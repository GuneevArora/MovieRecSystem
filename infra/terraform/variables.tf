variable "project_id" {
  description = "Google Cloud project to deploy into."
  type        = string
}

variable "region" {
  description = "Region for Cloud Run and Artifact Registry. us-central1 is a Tier 1 region with the full free allowance."
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Name of the Cloud Run service."
  type        = string
  default     = "movie-recommender-api"
}

variable "github_repository" {
  description = "GitHub repository allowed to deploy, as owner/name."
  type        = string
  default     = "GuneevArora/MovieRecSystem"
}

variable "allowed_origins" {
  description = "Browser origins allowed to call the API (CORS)."
  type        = list(string)
  default     = ["http://localhost:3000"]
}

variable "memory" {
  description = "Memory per instance. The service uses about 1 GB with the model loaded."
  type        = string
  default     = "2Gi"
}

variable "max_instances" {
  description = "Upper bound on instances. Kept low so a traffic spike cannot run up a bill."
  type        = number
  default     = 2
}
