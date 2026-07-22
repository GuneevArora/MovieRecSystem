output "service_url" {
  description = "Public URL of the API. Use it as NEXT_PUBLIC_API_URL in the web app."
  value       = google_cloud_run_v2_service.api.uri
}

# The values below go into GitHub repository variables for the deploy workflow.

output "GCP_PROJECT_ID" {
  value = var.project_id
}

output "GCP_REGION" {
  value = var.region
}

output "AR_REPOSITORY" {
  value = google_artifact_registry_repository.api.repository_id
}

output "CLOUD_RUN_SERVICE" {
  value = google_cloud_run_v2_service.api.name
}

output "GCP_WORKLOAD_IDENTITY_PROVIDER" {
  value = google_iam_workload_identity_pool_provider.github.name
}

output "GCP_DEPLOY_SERVICE_ACCOUNT" {
  value = google_service_account.deployer.email
}
