use std::collections::BTreeSet;

use aarambh_studio_core::{AarambhError, Result, TokenizerLike};
use aarambh_studio_tokenizer::{
    ASSISTANT_ID, BOS_ID, BpeTokenizer, PAD_ID, USER_ID, VIRTUAL_ASCII_END, tool_json_token_text,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::grammar::{JsonSchema, JsonSchemaGrammar, tool_call_schema};
use crate::thinking::{ThinkingController, ThinkingMode};

/// Marker selecting normal assistant text.
pub const FINAL_MARKER: &str = "<final>";
/// Marker opening a grammar-constrained tool call.
pub const TOOL_CALL_START: &str = "<tool_call>";
/// Marker closing a grammar-constrained tool call.
pub const TOOL_CALL_END: &str = "</tool_call>";

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Function definition exposed to the model.
pub struct ToolDefinition {
    /// Stable function name.
    pub name: String,
    /// Human-readable function behavior.
    #[serde(default)]
    pub description: String,
    /// JSON Schema describing the function arguments object.
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// One model-selected function call.
pub struct ToolCall {
    /// Selected function name.
    pub name: String,
    /// Schema-valid function arguments.
    pub arguments: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Policy controlling whether the model may call a tool.
pub enum ToolChoice {
    /// Let the model choose between a direct answer and one tool call.
    Auto,
    /// Disable tool calls.
    None,
    /// Require one of the supplied tools.
    Required,
    /// Require one specific tool by name.
    Named(String),
}

#[derive(Debug, Clone)]
struct CompiledTool {
    definition: ToolDefinition,
    schema: JsonSchema,
}

#[derive(Debug, Clone)]
/// Validated tool definitions and action-selection policy for generation.
pub struct ToolCallingConfig {
    tools: Vec<CompiledTool>,
    choice: ToolChoice,
}

impl ToolCallingConfig {
    /// Validate and compile tool definitions.
    pub fn new(definitions: Vec<ToolDefinition>, choice: ToolChoice) -> Result<Self> {
        if definitions.is_empty() {
            return Err(AarambhError::Config(
                "tool calling requires at least one tool".into(),
            ));
        }
        if definitions.len() > 64 {
            return Err(AarambhError::Config(
                "tool calling supports at most 64 tools per request".into(),
            ));
        }
        let mut names = BTreeSet::new();
        let mut tools = Vec::with_capacity(definitions.len());
        for definition in definitions {
            validate_tool_name(&definition.name)?;
            if !names.insert(definition.name.clone()) {
                return Err(AarambhError::Config(format!(
                    "duplicate tool name {:?}",
                    definition.name
                )));
            }
            let schema = JsonSchema::compile(&definition.parameters)?;
            if !schema.is_object() {
                return Err(AarambhError::Config(format!(
                    "tool {:?} parameters schema must have object type",
                    definition.name
                )));
            }
            tools.push(CompiledTool { definition, schema });
        }
        if let ToolChoice::Named(name) = &choice
            && !names.contains(name)
        {
            return Err(AarambhError::Config(format!(
                "named tool choice {name:?} is not present in tool definitions"
            )));
        }
        Ok(Self { tools, choice })
    }

    /// Return the configured tool-choice policy.
    pub fn choice(&self) -> &ToolChoice {
        &self.choice
    }

    /// Return validated tool definitions.
    pub fn definitions(&self) -> impl Iterator<Item = &ToolDefinition> {
        self.tools.iter().map(|tool| &tool.definition)
    }

    /// Render the deterministic chat prompt used by tool SFT and inference.
    pub fn render_prompt(&self, instruction: &str) -> Result<String> {
        let tools = self
            .tools
            .iter()
            .filter(|tool| match &self.choice {
                ToolChoice::Named(name) => tool.definition.name == *name,
                _ => true,
            })
            .map(|tool| {
                format!(
                    "{}: {}. Parameters: {}",
                    tool.definition.name,
                    tool.definition.description,
                    schema_summary(&tool.definition.parameters)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        Ok(format!(
            "<|user|>\nAvailable tools:\n{tools}\nChoose one tool when needed, otherwise answer directly.\nRequest: {instruction}\n<|assistant|>\n"
        ))
    }

    /// Validate a completed call against its selected tool definition.
    pub fn validate_call(&self, call: &ToolCall) -> Result<()> {
        let tool = self
            .tools
            .iter()
            .find(|tool| tool.definition.name == call.name)
            .ok_or_else(|| AarambhError::Config(format!("unknown tool call {:?}", call.name)))?;
        tool.schema.validate(&call.arguments)
    }

    fn grammar(&self) -> JsonSchemaGrammar {
        let alternatives = self
            .tools
            .iter()
            .filter(|tool| match &self.choice {
                ToolChoice::Named(name) => tool.definition.name == *name,
                _ => true,
            })
            .map(|tool| tool_call_schema(&tool.definition.name, &tool.schema))
            .collect();
        JsonSchemaGrammar::from_nodes(alternatives)
    }
}

fn schema_summary(schema: &Value) -> String {
    let Some(object) = schema.as_object() else {
        return "object".into();
    };
    let required = object
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<BTreeSet<_>>();
    object
        .get("properties")
        .and_then(Value::as_object)
        .map(|properties| {
            properties
                .iter()
                .map(|(name, schema)| {
                    let kind = schema
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or("value");
                    let suffix = if required.contains(name.as_str()) {
                        " required"
                    } else {
                        " optional"
                    };
                    format!("{name} {kind}{suffix}")
                })
                .collect::<Vec<_>>()
                .join(", ")
        })
        .filter(|summary| !summary.is_empty())
        .unwrap_or_else(|| "none".into())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolPhase {
    Thinking,
    Control,
    Answer,
    ToolCall,
}

#[derive(Debug, Clone)]
pub(crate) enum TokenConstraint {
    Any,
    Forced(u32),
    Allowed(Vec<u32>),
}

#[derive(Debug, Clone)]
enum ActionState {
    Select {
        final_tokens: Vec<u32>,
        tool_tokens: Vec<u32>,
        final_pos: Option<usize>,
        tool_pos: Option<usize>,
    },
    Answer,
    ToolJson(JsonSchemaGrammar),
    ToolClose {
        call: ToolCall,
        tokens: Vec<u32>,
        pos: usize,
    },
    Complete(ToolCall),
}

#[derive(Debug, Clone)]
/// Coordinates thinking, direct-answer selection, and constrained tool-call decoding.
pub struct ToolCallController {
    thinking: ThinkingController,
    config: ToolCallingConfig,
    state: ActionState,
}

impl ToolCallController {
    /// Build a controller for one generation request.
    pub fn new(
        mode: ThinkingMode,
        max_new_tokens: usize,
        config: ToolCallingConfig,
        tokenizer: &BpeTokenizer,
    ) -> Result<Self> {
        if tokenizer.vocab_size() <= VIRTUAL_ASCII_END as usize {
            return Err(AarambhError::Tokenizer(format!(
                "tool calling requires vocabulary size greater than {VIRTUAL_ASCII_END}"
            )));
        }
        let final_enabled = matches!(config.choice, ToolChoice::Auto | ToolChoice::None);
        let tool_enabled = !matches!(config.choice, ToolChoice::None);
        let final_tokens = if final_enabled {
            vec![ASSISTANT_ID, ASSISTANT_ID]
        } else {
            Vec::new()
        };
        let tool_tokens = if tool_enabled {
            vec![USER_ID, USER_ID]
        } else {
            Vec::new()
        };
        let final_pos = final_enabled.then_some(0);
        let tool_pos = tool_enabled.then_some(0);
        let state = ActionState::Select {
            final_tokens,
            tool_tokens,
            final_pos,
            tool_pos,
        };
        Ok(Self {
            thinking: ThinkingController::for_generation_with_reserve(mode, max_new_tokens, 128),
            config,
            state,
        })
    }

    pub(crate) fn constraint(&mut self, tokenizer: &BpeTokenizer) -> Result<TokenConstraint> {
        if let Some(force) = self.thinking.take_forced_token() {
            return Ok(TokenConstraint::Forced(force.token_id()));
        }
        if self.thinking.in_thinking_block()
            || (self.thinking.mode().is_enabled() && !self.thinking.is_closed())
        {
            return Ok(TokenConstraint::Any);
        }
        match &self.state {
            ActionState::Select {
                final_tokens,
                tool_tokens,
                final_pos,
                tool_pos,
            } => {
                let mut allowed = Vec::with_capacity(2);
                if let Some(pos) = final_pos {
                    allowed.push(final_tokens[*pos]);
                }
                if let Some(pos) = tool_pos {
                    allowed.push(tool_tokens[*pos]);
                }
                allowed.sort_unstable();
                allowed.dedup();
                match allowed.as_slice() {
                    [token] => Ok(TokenConstraint::Forced(*token)),
                    _ => Ok(TokenConstraint::Allowed(allowed)),
                }
            }
            ActionState::Answer => Ok(TokenConstraint::Any),
            ActionState::ToolJson(grammar) => Ok(TokenConstraint::Allowed(
                grammar.allowed_token_ids(tokenizer)?,
            )),
            ActionState::ToolClose { tokens, pos, .. } => Ok(TokenConstraint::Forced(tokens[*pos])),
            ActionState::Complete(_) => Err(AarambhError::Config(
                "tool controller requested a token after completion".into(),
            )),
        }
    }

    pub(crate) fn phase_for_next(&self) -> ToolPhase {
        if self.thinking.in_thinking_block()
            || (self.thinking.mode().is_enabled() && !self.thinking.is_closed())
        {
            ToolPhase::Thinking
        } else {
            match self.state {
                ActionState::Select { .. } | ActionState::ToolClose { .. } => ToolPhase::Control,
                ActionState::Answer => ToolPhase::Answer,
                ActionState::ToolJson(_) => ToolPhase::ToolCall,
                ActionState::Complete(_) => ToolPhase::Control,
            }
        }
    }

    pub(crate) fn on_token(
        &mut self,
        token_id: u32,
        _token_text: &str,
        tokenizer: &BpeTokenizer,
    ) -> Result<()> {
        if self.thinking.in_thinking_block()
            || (self.thinking.mode().is_enabled() && !self.thinking.is_closed())
        {
            self.thinking.on_token(token_id);
            return Ok(());
        }
        let mut transition = None;
        match &mut self.state {
            ActionState::Select {
                final_tokens,
                tool_tokens,
                final_pos,
                tool_pos,
            } => {
                advance_marker(final_pos, final_tokens, token_id);
                advance_marker(tool_pos, tool_tokens, token_id);
                if final_pos.is_none() && tool_pos.is_none() {
                    return Err(AarambhError::Config(
                        "action marker token matched no valid branch".into(),
                    ));
                }
                if final_pos.is_some_and(|pos| pos == final_tokens.len()) {
                    transition = Some(ActionState::Answer);
                } else if tool_pos.is_some_and(|pos| pos == tool_tokens.len()) {
                    transition = Some(ActionState::ToolJson(self.config.grammar()));
                }
            }
            ActionState::Answer => {}
            ActionState::ToolJson(grammar) => {
                grammar.accept_token_id(token_id, tokenizer)?;
                if grammar.is_complete() {
                    let value = grammar.finish()?;
                    let call: ToolCall = serde_json::from_value(value)?;
                    self.config.validate_call(&call)?;
                    transition = Some(ActionState::ToolClose {
                        call,
                        tokens: vec![BOS_ID, PAD_ID],
                        pos: 0,
                    });
                }
            }
            ActionState::ToolClose { call, tokens, pos } => {
                if tokens[*pos] != token_id {
                    return Err(AarambhError::Config(
                        "tool-call closing marker token mismatch".into(),
                    ));
                }
                *pos += 1;
                if *pos == tokens.len() {
                    transition = Some(ActionState::Complete(call.clone()));
                }
            }
            ActionState::Complete(_) => {}
        }
        if let Some(state) = transition {
            self.state = state;
        }
        Ok(())
    }

    pub(crate) fn is_complete(&self) -> bool {
        matches!(self.state, ActionState::Complete(_))
    }

    /// Return the completed tool call, if one was selected.
    pub fn tool_call(&self) -> Option<&ToolCall> {
        match &self.state {
            ActionState::Complete(call) => Some(call),
            _ => None,
        }
    }

    pub(crate) fn action_is_resolved(&self) -> bool {
        matches!(self.state, ActionState::Answer | ActionState::Complete(_))
    }

    pub(crate) fn thinking(&self) -> &ThinkingController {
        &self.thinking
    }

    pub(crate) fn token_text(&self, token_id: u32, tokenizer: &BpeTokenizer) -> Result<String> {
        match self.phase_for_next() {
            ToolPhase::Control => Ok(String::new()),
            ToolPhase::ToolCall => tool_json_token_text(token_id, tokenizer),
            ToolPhase::Thinking | ToolPhase::Answer => tokenizer.decode(&[token_id]),
        }
    }
}

fn advance_marker(position: &mut Option<usize>, tokens: &[u32], token_id: u32) {
    let Some(pos) = *position else {
        return;
    };
    if tokens.get(pos).copied() == Some(token_id) {
        *position = Some(pos + 1);
    } else {
        *position = None;
    }
}

fn validate_tool_name(name: &str) -> Result<()> {
    let mut chars = name.chars();
    let first = chars
        .next()
        .ok_or_else(|| AarambhError::Config("tool name must not be empty".into()))?;
    if name.len() > 64
        || !(first.is_ascii_alphabetic() || first == '_')
        || !chars.all(|char| char.is_ascii_alphanumeric() || matches!(char, '_' | '.' | '-'))
    {
        return Err(AarambhError::Config(format!(
            "invalid tool name {name:?}; expected [A-Za-z_][A-Za-z0-9_.-]{{0,63}}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aarambh_studio_tokenizer::{
        ASSISTANT, ASSISTANT_ID, BOS, BOS_ID, ENDOFTEXT, ENDOFTEXT_ID, PAD, PAD_ID, THINK_END,
        THINK_END_ID, THINK_START, THINK_START_ID, USER, USER_ID, Vocab,
    };
    use std::collections::HashMap;

    #[test]
    fn duplicate_tool_names_are_rejected() {
        let tool = ToolDefinition {
            name: "weather".into(),
            description: String::new(),
            parameters: serde_json::json!({"type":"object","properties":{}}),
        };
        assert!(ToolCallingConfig::new(vec![tool.clone(), tool], ToolChoice::Auto).is_err());
    }

    #[test]
    fn call_validation_uses_selected_schema() {
        let config = ToolCallingConfig::new(
            vec![ToolDefinition {
                name: "weather".into(),
                description: String::new(),
                parameters: serde_json::json!({
                    "type":"object",
                    "properties":{"city":{"type":"string"}},
                    "required":["city"]
                }),
            }],
            ToolChoice::Required,
        )
        .unwrap();
        config
            .validate_call(&ToolCall {
                name: "weather".into(),
                arguments: serde_json::json!({"city":"Delhi"}),
            })
            .unwrap();
    }

    #[test]
    fn required_call_reaches_typed_completion() {
        let tokenizer = character_tokenizer();
        let config = ToolCallingConfig::new(
            vec![ToolDefinition {
                name: "weather".into(),
                description: String::new(),
                parameters: serde_json::json!({
                    "type":"object",
                    "properties":{"city":{"type":"string"}},
                    "required":["city"]
                }),
            }],
            ToolChoice::Required,
        )
        .unwrap();
        let mut controller =
            ToolCallController::new(ThinkingMode::None, 128, config, &tokenizer).unwrap();
        let mut target = vec![USER_ID, USER_ID];
        target.extend(virtual_json_ids(
            r#"{"name":"weather","arguments":{"city":"Delhi"}}"#,
            &tokenizer,
        ));
        target.extend([BOS_ID, PAD_ID]);
        for token_id in target {
            let constraint = controller.constraint(&tokenizer).unwrap();
            match constraint {
                TokenConstraint::Any => {}
                TokenConstraint::Forced(expected) => assert_eq!(expected, token_id),
                TokenConstraint::Allowed(allowed) => assert!(allowed.contains(&token_id)),
            }
            let text = tokenizer.decode(&[token_id]).unwrap();
            controller.on_token(token_id, &text, &tokenizer).unwrap();
        }
        assert!(controller.is_complete());
        assert_eq!(
            controller.tool_call(),
            Some(&ToolCall {
                name: "weather".into(),
                arguments: serde_json::json!({"city":"Delhi"}),
            })
        );
    }

    #[test]
    fn no_tool_choice_enters_normal_answer_after_control_tokens() {
        let tokenizer = character_tokenizer();
        let config = ToolCallingConfig::new(
            vec![ToolDefinition {
                name: "weather".into(),
                description: String::new(),
                parameters: serde_json::json!({"type":"object","properties":{}}),
            }],
            ToolChoice::None,
        )
        .unwrap();
        let mut controller =
            ToolCallController::new(ThinkingMode::None, 16, config, &tokenizer).unwrap();
        for token_id in [ASSISTANT_ID, ASSISTANT_ID] {
            assert!(matches!(
                controller.constraint(&tokenizer).unwrap(),
                TokenConstraint::Forced(expected) if expected == token_id
            ));
            controller.on_token(token_id, "", &tokenizer).unwrap();
        }
        assert_eq!(controller.phase_for_next(), ToolPhase::Answer);
        assert!(matches!(
            controller.constraint(&tokenizer).unwrap(),
            TokenConstraint::Any
        ));
    }

    #[test]
    fn thinking_closes_before_tool_selection() {
        let tokenizer = character_tokenizer();
        let config = ToolCallingConfig::new(
            vec![ToolDefinition {
                name: "weather".into(),
                description: String::new(),
                parameters: serde_json::json!({"type":"object","properties":{}}),
            }],
            ToolChoice::Required,
        )
        .unwrap();
        let mut controller =
            ToolCallController::new(ThinkingMode::Low, 16, config, &tokenizer).unwrap();
        assert!(matches!(
            controller.constraint(&tokenizer).unwrap(),
            TokenConstraint::Forced(THINK_START_ID)
        ));
        controller
            .on_token(THINK_START_ID, THINK_START, &tokenizer)
            .unwrap();
        assert!(matches!(
            controller.constraint(&tokenizer).unwrap(),
            TokenConstraint::Forced(THINK_END_ID)
        ));
        controller
            .on_token(THINK_END_ID, THINK_END, &tokenizer)
            .unwrap();
        assert_eq!(controller.phase_for_next(), ToolPhase::Control);
    }

    fn character_tokenizer() -> BpeTokenizer {
        let mut token_to_id = HashMap::from([
            (ENDOFTEXT.to_string(), ENDOFTEXT_ID),
            (PAD.to_string(), PAD_ID),
            (BOS.to_string(), BOS_ID),
            (THINK_START.to_string(), THINK_START_ID),
            (THINK_END.to_string(), THINK_END_ID),
            (USER.to_string(), USER_ID),
            (ASSISTANT.to_string(), ASSISTANT_ID),
        ]);
        let mut id_to_token = vec![String::new(); 7];
        for (token, id) in &token_to_id {
            id_to_token[*id as usize] = token.clone();
        }
        while id_to_token.len() <= VIRTUAL_ASCII_END as usize {
            let id = id_to_token.len() as u32;
            let token = format!("<reserved_{id}>");
            token_to_id.insert(token.clone(), id);
            id_to_token.push(token);
        }
        for character in "<tool_call>/>{\"name:weather,gumtscityDelhi}".chars() {
            let token = character.to_string();
            if token_to_id.contains_key(&token) {
                continue;
            }
            let id = id_to_token.len() as u32;
            token_to_id.insert(token.clone(), id);
            id_to_token.push(token);
        }
        BpeTokenizer {
            vocab: Vocab {
                token_to_id,
                id_to_token,
            },
            merges: Vec::new(),
            merge_rank: HashMap::new(),
            chat_template_version: None,
        }
    }

    fn virtual_json_ids(text: &str, tokenizer: &BpeTokenizer) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut regular = String::new();
        for character in text.chars() {
            let structural = match character {
                '{' => Some(BOS_ID),
                '}' => Some(PAD_ID),
                '[' => Some(THINK_START_ID),
                ']' => Some(THINK_END_ID),
                '"' => Some(USER_ID),
                ':' => Some(ASSISTANT_ID),
                ',' => Some(ENDOFTEXT_ID),
                _ => None,
            };
            if let Some(token_id) = structural {
                if !regular.is_empty() {
                    ids.extend(tokenizer.encode(&regular).unwrap());
                    regular.clear();
                }
                ids.push(token_id);
            } else {
                regular.push(character);
            }
        }
        if !regular.is_empty() {
            ids.extend(tokenizer.encode(&regular).unwrap());
        }
        ids
    }
}
