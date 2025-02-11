use datafusion::common::extensions_options;
use datafusion::config::ConfigExtension;

extensions_options! {
    /// LLM configuration options.
    pub struct LLMConfig {
        /// The name of the model to use.
        pub model: String, default = "default_model".to_string()
        /// The endpoint for the LLM chat API.
        pub chat_endpoint: String, default = "http://localhost:8080/chat".to_string()
        /// The API key to use for the LLM chat API.
        /// TODO: It's better to use Option but it's not supported yet.
        pub api_key: String, default = "".to_string()
    }
}

impl ConfigExtension for LLMConfig {
    const PREFIX: &'static str = "llm";
}
