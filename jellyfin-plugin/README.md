# MovieRec for Jellyfin

A Jellyfin server plugin that turns each user's watch history into recommendations, using the API in this repository.

## What it does

- **A playlist per user.** A scheduled task (daily at 04:00, or run it by hand under Dashboard → Scheduled Tasks) reads the films a user has played or favourited, asks the API to rank the unwatched films in the library for them, and keeps a private playlist called "Recommended for you" up to date. Playlists show in every Jellyfin client, so nothing has to be installed on TVs or phones.
- **Films you don't have.** The plugin's settings page has a preview that also lists the best-ranked films missing from the library. With Jellyseerr configured, each has a Request button.

Films are matched by TMDB id, so the library needs TMDB metadata (Jellyfin's default for films). The model reads history in the order films were last played.

## Requirements

- Jellyfin **10.11.x**. A plugin only loads on the server line it was built for; to build for another line change `TargetFramework` and `JellyfinVersion` in the `.csproj` (10.10 uses `net8.0`).
- The recommender API reachable **from the Jellyfin server**. Jellyfin itself does not need to be public: the plugin only makes outgoing requests. Either point it at a hosted API, or run the API's Docker image on the same machine or network.
- Optional: Jellyseerr and its API key.

## Build and install

```bash
dotnet build jellyfin-plugin/Jellyfin.Plugin.MovieRec -c Release
```

1. Create a folder `MovieRec` inside the server's `plugins` directory (for Docker installs that is `/config/plugins`).
2. Copy `bin/Release/net9.0/Jellyfin.Plugin.MovieRec.dll` into it.
3. Restart Jellyfin.
4. Open Dashboard → Plugins → MovieRec, set the API URL, save, and press "Show my recommendations".

CI also builds the DLL and attaches it to each run as the `jellyfin-plugin-movierec` artifact.

## Endpoints added to Jellyfin

| Endpoint | Who | Purpose |
|---|---|---|
| `GET /MovieRec/Recommendations` | Any signed-in user, for themselves; administrators may pass `userId` | Library picks and missing films |
| `POST /MovieRec/Request/{tmdbId}` | Administrators | Request a film through Jellyseerr |

Requests are limited to administrators because the plugin calls Jellyseerr with its API key, which has full rights there.

## Privacy

The plugin sends lists of TMDB ids (watched films and library contents) to the API URL you configure. No usernames, user ids or file paths leave the server.

## Licence

The source here is MIT, like the rest of the repository. A compiled plugin links against Jellyfin's libraries, which are GPL-3.0, so a built DLL is distributed under GPL-3.0 terms.
