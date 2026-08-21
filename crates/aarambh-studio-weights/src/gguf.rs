use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use aarambh_studio_core::{AarambhError, Configurable, ModelConfig, Result};
use aarambh_studio_model::AarambhModel;
use aarambh_studio_quant::{
    GgufFormat, Q4_K_M_BLOCK_SIZE, Q4_K_M_ENCODED_SIZE, dequantise_block_q4_k_m, dequantise_i8,
    f16_to_f32, f32_to_f16, quantise_absmax_i8, quantise_block_q4_k_m,
};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};

const MAGIC: &[u8; 4] = b"GGUF";
const VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GgufMetadata {
    config: ModelConfig,
    format: GgufFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TensorEncoding {
    F32 = 0,
    Q8_0 = 1,
    Q4KM = 2,
}

impl TensorEncoding {
    fn from_u8(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::Q8_0),
            2 => Ok(Self::Q4KM),
            other => Err(AarambhError::Checkpoint(format!(
                "unknown GGUF tensor encoding {other}"
            ))),
        }
    }
}

/// Save an Aarambh model to the v1 GGUF-compatible container.
pub fn save_gguf(model: &AarambhModel, format: GgufFormat, path: impl AsRef<Path>) -> Result<()> {
    let file = File::create(path.as_ref())?;
    let mut writer = BufWriter::new(file);
    writer.write_all(MAGIC)?;
    write_u32(&mut writer, VERSION)?;
    let metadata = serde_json::to_vec(&GgufMetadata {
        config: model.config().clone(),
        format,
    })?;
    write_u64(&mut writer, metadata.len() as u64)?;
    writer.write_all(&metadata)?;

    let mut tensors = model.named_tensors().into_iter().collect::<Vec<_>>();
    tensors.sort_by(|(a, _), (b, _)| a.cmp(b));
    write_u64(&mut writer, tensors.len() as u64)?;
    for (name, tensor) in tensors {
        write_string(&mut writer, &name)?;
        let shape = tensor.dims().to_vec();
        write_shape(&mut writer, &shape)?;
        let (encoding, payload) = encode_tensor(&name, &tensor, format)?;
        writer.write_all(&[encoding as u8])?;
        write_u64(&mut writer, payload.len() as u64)?;
        writer.write_all(&payload)?;
    }
    writer.flush()?;
    Ok(())
}

/// Load a v1 GGUF-compatible checkpoint using f32 parameters.
pub fn load_gguf(path: impl AsRef<Path>, device: &Device) -> Result<AarambhModel> {
    load_gguf_with_dtype(path, device, DType::F32)
}

/// Load a v1 GGUF-compatible checkpoint using the requested dtype.
pub fn load_gguf_with_dtype(
    path: impl AsRef<Path>,
    device: &Device,
    dtype: DType,
) -> Result<AarambhModel> {
    let (config, tensors) = load_gguf_tensors(path, device)?;
    let vb = VarBuilder::from_tensors(tensors, dtype, device);
    AarambhModel::new(&config, vb)
}

/// Load raw tensors and model configuration from a v1 GGUF-compatible checkpoint.
pub fn load_gguf_tensors(
    path: impl AsRef<Path>,
    device: &Device,
) -> Result<(ModelConfig, HashMap<String, Tensor>)> {
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file);
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(AarambhError::Checkpoint(format!(
            "{} is not an Aarambh GGUF file",
            path.as_ref().display()
        )));
    }
    let version = read_u32(&mut reader)?;
    if version != VERSION {
        return Err(AarambhError::Checkpoint(format!(
            "unsupported GGUF version {version}, expected {VERSION}"
        )));
    }
    let metadata_len = read_u64(&mut reader)? as usize;
    let mut metadata_bytes = vec![0u8; metadata_len];
    reader.read_exact(&mut metadata_bytes)?;
    let metadata: GgufMetadata = serde_json::from_slice(&metadata_bytes)?;

    let tensor_count = read_u64(&mut reader)? as usize;
    let mut tensors = HashMap::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = read_string(&mut reader)?;
        let shape = read_shape(&mut reader)?;
        let mut encoding = [0u8; 1];
        reader.read_exact(&mut encoding)?;
        let encoding = TensorEncoding::from_u8(encoding[0])?;
        let payload_len = read_u64(&mut reader)? as usize;
        let mut payload = vec![0u8; payload_len];
        reader.read_exact(&mut payload)?;
        let tensor = decode_tensor(encoding, &shape, &payload, device)?;
        tensors.insert(name, tensor);
    }
    Ok((metadata.config, tensors))
}

fn encode_tensor(
    name: &str,
    tensor: &Tensor,
    format: GgufFormat,
) -> Result<(TensorEncoding, Vec<u8>)> {
    if tensor.dims().len() != 2 || name.contains("_conv.") {
        return Ok((TensorEncoding::F32, encode_f32_tensor(tensor)?));
    }
    if name.contains(".dsa.index_") {
        return Ok((TensorEncoding::Q8_0, encode_q8_tensor(tensor)?));
    }
    match format {
        GgufFormat::Q80 => Ok((TensorEncoding::Q8_0, encode_q8_tensor(tensor)?)),
        GgufFormat::Q4KM | GgufFormat::Q5KM => {
            Ok((TensorEncoding::Q4KM, encode_q4_k_m_tensor(tensor)?))
        }
    }
}

fn decode_tensor(
    encoding: TensorEncoding,
    shape: &[usize],
    payload: &[u8],
    device: &Device,
) -> Result<Tensor> {
    match encoding {
        TensorEncoding::F32 => decode_f32_tensor(shape, payload, device),
        TensorEncoding::Q8_0 => decode_q8_tensor(shape, payload, device),
        TensorEncoding::Q4KM => decode_q4_k_m_tensor(shape, payload, device),
    }
}

fn encode_f32_tensor(tensor: &Tensor) -> Result<Vec<u8>> {
    let values = tensor.flatten_all()?.to_vec1::<f32>()?;
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

fn decode_f32_tensor(shape: &[usize], payload: &[u8], device: &Device) -> Result<Tensor> {
    if !payload.len().is_multiple_of(4) {
        return Err(AarambhError::Checkpoint(
            "invalid F32 GGUF payload length".into(),
        ));
    }
    let values = payload
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect::<Vec<_>>();
    Ok(Tensor::from_vec(values, shape, device)?)
}

fn encode_q8_tensor(tensor: &Tensor) -> Result<Vec<u8>> {
    let quantized = quantise_absmax_i8(tensor)?;
    let mut bytes = Vec::with_capacity(4 + quantized.data.len());
    bytes.extend_from_slice(&quantized.scale.to_le_bytes());
    bytes.extend(quantized.data.into_iter().map(|value| value as u8));
    Ok(bytes)
}

fn decode_q8_tensor(shape: &[usize], payload: &[u8], device: &Device) -> Result<Tensor> {
    if payload.len() < 4 {
        return Err(AarambhError::Checkpoint("invalid Q8_0 payload".into()));
    }
    let scale = f32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
    let data = payload[4..]
        .iter()
        .map(|value| *value as i8)
        .collect::<Vec<_>>();
    dequantise_i8(
        &aarambh_studio_quant::I8QuantizedTensor {
            shape: shape.to_vec(),
            data,
            scale,
        },
        device,
    )
}

fn encode_q4_k_m_tensor(tensor: &Tensor) -> Result<Vec<u8>> {
    let values = tensor.flatten_all()?.to_vec1::<f32>()?;
    let mut bytes =
        Vec::with_capacity(values.len().div_ceil(Q4_K_M_BLOCK_SIZE) * Q4_K_M_ENCODED_SIZE);
    for chunk in values.chunks(Q4_K_M_BLOCK_SIZE) {
        let mut block = [0.0f32; Q4_K_M_BLOCK_SIZE];
        block[..chunk.len()].copy_from_slice(chunk);
        bytes.extend_from_slice(&quantise_block_q4_k_m(&block));
    }
    Ok(bytes)
}

fn decode_q4_k_m_tensor(shape: &[usize], payload: &[u8], device: &Device) -> Result<Tensor> {
    if !payload.len().is_multiple_of(Q4_K_M_ENCODED_SIZE) {
        return Err(AarambhError::Checkpoint(
            "invalid Q4_K_M GGUF payload length".into(),
        ));
    }
    let target_len = shape.iter().product::<usize>();
    let mut values = Vec::with_capacity(payload.len() / Q4_K_M_ENCODED_SIZE * Q4_K_M_BLOCK_SIZE);
    for chunk in payload.as_chunks::<Q4_K_M_ENCODED_SIZE>().0 {
        let mut block = [0u8; Q4_K_M_ENCODED_SIZE];
        block.copy_from_slice(chunk);
        values.extend_from_slice(&dequantise_block_q4_k_m(&block));
    }
    values.truncate(target_len);
    Ok(Tensor::from_vec(values, shape, device)?)
}

/// Encode two f32 values as adjacent f16 little-endian values.
pub fn encode_f16_pair(scale: f32, min: f32) -> [u8; 4] {
    let mut bytes = [0u8; 4];
    bytes[0..2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
    bytes[2..4].copy_from_slice(&f32_to_f16(min).to_le_bytes());
    bytes
}

/// Decode two adjacent f16 little-endian values to f32.
pub fn decode_f16_pair(bytes: [u8; 4]) -> (f32, f32) {
    (
        f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])),
        f16_to_f32(u16::from_le_bytes([bytes[2], bytes[3]])),
    )
}

fn write_string(writer: &mut impl Write, value: &str) -> Result<()> {
    write_u64(writer, value.len() as u64)?;
    writer.write_all(value.as_bytes())?;
    Ok(())
}

fn read_string(reader: &mut impl Read) -> Result<String> {
    let len = read_u64(reader)? as usize;
    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|err| AarambhError::Checkpoint(err.to_string()))
}

fn write_shape(writer: &mut impl Write, shape: &[usize]) -> Result<()> {
    write_u64(writer, shape.len() as u64)?;
    for dim in shape {
        write_u64(writer, *dim as u64)?;
    }
    Ok(())
}

fn read_shape(reader: &mut impl Read) -> Result<Vec<usize>> {
    let n_dims = read_u64(reader)? as usize;
    let mut shape = Vec::with_capacity(n_dims);
    for _ in 0..n_dims {
        shape.push(read_u64(reader)? as usize);
    }
    Ok(shape)
}

fn write_u32(writer: &mut impl Write, value: u32) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn write_u64(writer: &mut impl Write, value: u64) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn read_u64(reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn f16_pair_roundtrip_is_finite() {
        let bytes = encode_f16_pair(0.125, -2.5);
        let (scale, min) = decode_f16_pair(bytes);
        assert!((scale - 0.125).abs() < 0.001);
        assert!((min + 2.5).abs() < 0.001);
    }

    #[test]
    fn q4_payload_roundtrip_shape() {
        let device = Device::Cpu;
        let tensor = Tensor::from_vec(
            (0..300)
                .map(|idx| idx as f32 * 0.01 - 1.5)
                .collect::<Vec<_>>(),
            (20, 15),
            &device,
        )
        .unwrap();
        let payload = encode_q4_k_m_tensor(&tensor).unwrap();
        let decoded = decode_q4_k_m_tensor(tensor.dims(), &payload, &device).unwrap();
        assert_eq!(decoded.dims(), tensor.dims());
    }
}
