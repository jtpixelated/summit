use super::api_client::{ApiClient, AuthMethod};
use super::base::{ConfigKey, ProviderDef, ProviderMetadata};
use super::openai_compatible::OpenAiCompatibleProvider;
use crate::model::ModelConfig;
use anyhow::Result;
use futures::future::BoxFuture;

const GROQ_PROVIDER_NAME: &str = "groq";
pub const GROQ_API_HOST: &str = "https://api.groq.com/openai/v1";

/// Act / default model — strong coding ability, low cost.
pub const GROQ_MODEL_DEFAULT: &str = "llama-3.3-70b-versatile";

/// Fast model — used for Reflect, Summarize, and Rush mode.
pub const GROQ_MODEL_FAST: &str = "llama-3.1-8b-instant";

/// Deep model — used for Plan in Deep mode (extended context, strongest reasoning on Groq).
pub const GROQ_MODEL_DEEP: &str = "openai/gpt-oss-120b";

/// Compound model — server-side web search, code execution, visit_website.
pub const GROQ_MODEL_COMPOUND: &str = "compound-beta";

pub const GROQ_KNOWN_MODELS: &[&str] = &[
    GROQ_MODEL_FAST,
    GROQ_MODEL_DEFAULT,
    GROQ_MODEL_DEEP,
    GROQ_MODEL_COMPOUND,
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
    "moonshotai/kimi-k2-instruct-0905",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-guard-4-12b",
];

pub const GROQ_DOC_URL: &str = "https://console.groq.com/docs/openai";

pub struct GroqProvider;

impl ProviderDef for GroqProvider {
    type Provider = OpenAiCompatibleProvider;

    fn metadata() -> ProviderMetadata {
        ProviderMetadata::new(
            GROQ_PROVIDER_NAME,
            "Groq",
            "Sub-second inference on Groq LPU hardware — fast, cheap, ideal for coding tasks",
            GROQ_MODEL_DEFAULT,
            GROQ_KNOWN_MODELS.to_vec(),
            GROQ_DOC_URL,
            vec![
                ConfigKey::new("GROQ_API_KEY", true, true, None, true),
                ConfigKey::new("GROQ_HOST", false, false, Some(GROQ_API_HOST), false),
            ],
        )
    }

    fn from_env(
        model: ModelConfig,
        _extensions: Vec<crate::config::ExtensionConfig>,
    ) -> BoxFuture<'static, Result<OpenAiCompatibleProvider>> {
        Box::pin(async move {
            let config = crate::config::Config::global();
            let api_key: String = config.get_secret("GROQ_API_KEY")?;
            let host: String = config
                .get_param("GROQ_HOST")
                .unwrap_or_else(|_| GROQ_API_HOST.to_string());

            let api_client = ApiClient::new(host, AuthMethod::BearerToken(api_key))?
                // Enable Groq prompt caching on every request.
                // Cache hit metrics are logged via x-groq-prompt-cache-hit response headers.
                .with_header("x-groq-prompt-cache", "enabled")?;

            Ok(OpenAiCompatibleProvider::new(
                GROQ_PROVIDER_NAME.to_string(),
                api_client,
                model,
                String::new(),
            ))
        })
    }
}
