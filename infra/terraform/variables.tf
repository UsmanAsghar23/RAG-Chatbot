variable "aws_region" {
  description = "AWS region for all resources."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Short project name used in resource names."
  type        = string
  default     = "rag-api"
}

variable "environment" {
  description = "Deployment environment label."
  type        = string
  default     = "prod"
}

variable "container_port" {
  description = "Port exposed by the FastAPI container."
  type        = number
  default     = 8000
}

variable "desired_count" {
  description = "Number of ECS tasks to run."
  type        = number
  default     = 2
}

variable "task_cpu" {
  description = "Fargate task CPU units."
  type        = number
  default     = 512
}

variable "task_memory" {
  description = "Fargate task memory in MiB."
  type        = number
  default     = 1024
}

variable "certificate_arn" {
  description = "Optional ACM certificate ARN for HTTPS on the ALB."
  type        = string
  default     = ""
}

variable "openai_api_key" {
  description = "OpenAI API key stored in Secrets Manager."
  type        = string
  sensitive   = true
}

variable "pinecone_api_key" {
  description = "Pinecone API key stored in Secrets Manager."
  type        = string
  sensitive   = true
}

variable "api_key" {
  description = "Application API key stored in Secrets Manager."
  type        = string
  sensitive   = true
}

variable "pinecone_index_name" {
  description = "Pinecone index name."
  type        = string
  default     = "rag-index"
}

variable "embedding_model" {
  description = "OpenAI embedding model."
  type        = string
  default     = "text-embedding-3-small"
}

variable "chat_model" {
  description = "OpenAI chat model."
  type        = string
  default     = "gpt-4o"
}

variable "chunk_size" {
  description = "Document chunk size in tokens."
  type        = number
  default     = 512
}

variable "chunk_overlap" {
  description = "Chunk overlap in tokens."
  type        = number
  default     = 64
}

variable "top_k" {
  description = "Default retrieval top_k."
  type        = number
  default     = 5
}

variable "max_upload_mb" {
  description = "Maximum upload size in megabytes."
  type        = number
  default     = 10
}

variable "app_name" {
  description = "FastAPI application title."
  type        = string
  default     = "RAG Chatbot"
}

variable "image_tag" {
  description = "Docker image tag deployed to ECS."
  type        = string
  default     = "latest"
}
