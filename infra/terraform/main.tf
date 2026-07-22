locals {
  apis = [
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "sts.googleapis.com",
  ]

  # Shown by Cloud Run until the first real image is deployed by CI.
  placeholder_image = "us-docker.pkg.dev/cloudrun/container/hello"
}

data "google_project" "this" {
  project_id = var.project_id
}

resource "google_project_service" "apis" {
  for_each = toset(local.apis)

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# ---------------------------------------------------------------------------
# Container registry
# ---------------------------------------------------------------------------

resource "google_artifact_registry_repository" "api" {
  project       = var.project_id
  location      = var.region
  repository_id = var.service_name
  description   = "Images for the movie recommender API"
  format        = "DOCKER"

  # Keep storage near the free allowance: only the two newest images survive.
  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"

    most_recent_versions {
      keep_count = 2
    }
  }

  cleanup_policies {
    id     = "delete-old"
    action = "DELETE"

    condition {
      older_than = "86400s"
    }
  }

  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------------------
# Cloud Run service
# ---------------------------------------------------------------------------

resource "google_service_account" "runtime" {
  project      = var.project_id
  account_id   = "${var.service_name}-run"
  display_name = "Movie recommender API runtime"
}

resource "google_cloud_run_v2_service" "api" {
  project             = var.project_id
  name                = var.service_name
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account = google_service_account.runtime.email

    # One request-batching process per instance handles many requests at once.
    max_instance_request_concurrency = 80
    timeout                          = "30s"

    scaling {
      min_instance_count = 0
      max_instance_count = var.max_instances
    }

    containers {
      image = local.placeholder_image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "1"
          memory = var.memory
        }

        # Only bill CPU while a request is being served.
        cpu_idle          = true
        startup_cpu_boost = true
      }

      env {
        name  = "ALLOWED_ORIGINS"
        value = join(",", var.allowed_origins)
      }

      env {
        name  = "TORCH_THREADS"
        value = "1"
      }

      startup_probe {
        initial_delay_seconds = 0
        period_seconds        = 3
        timeout_seconds       = 2
        failure_threshold     = 40

        http_get {
          path = "/health"
        }
      }
    }
  }

  lifecycle {
    # CI owns the image tag after the first deploy.
    ignore_changes = [
      template[0].containers[0].image,
      client,
      client_version,
    ]
  }

  depends_on = [google_project_service.apis]
}

# The demo API is public and unauthenticated by design.
resource "google_cloud_run_v2_service_iam_member" "public" {
  project  = var.project_id
  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ---------------------------------------------------------------------------
# Keyless deploys from GitHub Actions (Workload Identity Federation)
# ---------------------------------------------------------------------------

resource "google_service_account" "deployer" {
  project      = var.project_id
  account_id   = "${var.service_name}-ci"
  display_name = "GitHub Actions deployer"
}

resource "google_iam_workload_identity_pool" "github" {
  project                   = var.project_id
  workload_identity_pool_id = "github-actions"
  display_name              = "GitHub Actions"

  depends_on = [google_project_service.apis]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  project                            = var.project_id
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github"
  display_name                       = "GitHub OIDC"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
    "attribute.ref"        = "assertion.ref"
  }

  # Only tokens issued to this repository are accepted.
  attribute_condition = "assertion.repository == \"${var.github_repository}\""

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account_iam_member" "github_impersonates_deployer" {
  service_account_id = google_service_account.deployer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repository}"
}

resource "google_artifact_registry_repository_iam_member" "deployer_pushes_images" {
  project    = var.project_id
  location   = google_artifact_registry_repository.api.location
  repository = google_artifact_registry_repository.api.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.deployer.email}"
}

resource "google_cloud_run_v2_service_iam_member" "deployer_updates_service" {
  project  = var.project_id
  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.developer"
  member   = "serviceAccount:${google_service_account.deployer.email}"
}

# Deploying a revision requires permission to act as its runtime identity.
resource "google_service_account_iam_member" "deployer_uses_runtime" {
  service_account_id = google_service_account.runtime.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.deployer.email}"
}
