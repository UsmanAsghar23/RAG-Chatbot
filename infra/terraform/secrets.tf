resource "aws_secretsmanager_secret" "app" {
  name = "${local.name_prefix}/app-secrets"

  tags = local.common_tags
}

resource "aws_secretsmanager_secret_version" "app" {
  secret_id = aws_secretsmanager_secret.app.id

  secret_string = jsonencode({
    OPENAI_API_KEY   = var.openai_api_key
    PINECONE_API_KEY = var.pinecone_api_key
    API_KEY          = var.api_key
  })
}
