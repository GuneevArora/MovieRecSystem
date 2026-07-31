using System.Net.Http.Json;
using System.Text.Json.Serialization;
using MediaBrowser.Common.Net;

namespace Jellyfin.Plugin.MovieRec;

/// <summary>A film as returned by the recommender API.</summary>
public class ApiMovie
{
    [JsonPropertyName("tmdb_id")]
    public int? TmdbId { get; set; }

    [JsonPropertyName("title")]
    public string? Title { get; set; }

    [JsonPropertyName("year")]
    public int? Year { get; set; }

    [JsonPropertyName("genres")]
    public List<string> Genres { get; set; } = [];

    [JsonPropertyName("poster_path")]
    public string? PosterPath { get; set; }

    [JsonPropertyName("overview")]
    public string? Overview { get; set; }

    [JsonPropertyName("score")]
    public double? Score { get; set; }

    [JsonPropertyName("cold_start")]
    public bool ColdStart { get; set; }
}

/// <summary>Response of POST /recommend/tmdb.</summary>
public class ApiRecommendations
{
    [JsonPropertyName("items")]
    public List<ApiMovie> Items { get; set; } = [];

    [JsonPropertyName("strategy")]
    public string Strategy { get; set; } = string.Empty;

    [JsonPropertyName("unknown_ids")]
    public List<int> UnknownIds { get; set; } = [];
}

/// <summary>
/// Talks to the recommender API and to Jellyseerr.
/// </summary>
public class RecommenderClient
{
    private readonly IHttpClientFactory _httpClientFactory;

    public RecommenderClient(IHttpClientFactory httpClientFactory)
    {
        _httpClientFactory = httpClientFactory;
    }

    /// <summary>
    /// Ranks films for a watch history given as TMDB ids, oldest first.
    /// </summary>
    /// <param name="history">Liked films, oldest first.</param>
    /// <param name="k">How many films to return.</param>
    /// <param name="candidates">If not null, only these films are ranked.</param>
    /// <param name="exclude">Films to leave out.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The ranked films.</returns>
    public async Task<ApiRecommendations> RecommendAsync(
        IReadOnlyList<int> history,
        int k,
        IReadOnlyCollection<int>? candidates,
        IReadOnlyCollection<int>? exclude,
        CancellationToken cancellationToken)
    {
        var body = new Dictionary<string, object?>
        {
            ["tmdb_ids"] = history,
            ["k"] = k,
            ["candidate_tmdb_ids"] = candidates,
            ["exclude_tmdb_ids"] = exclude ?? [],
        };

        var client = _httpClientFactory.CreateClient(NamedClient.Default);
        using var response = await client
            .PostAsJsonAsync(BuildUrl(Config.ApiUrl, "recommend/tmdb"), body, cancellationToken)
            .ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        return await response.Content
            .ReadFromJsonAsync<ApiRecommendations>(cancellationToken)
            .ConfigureAwait(false) ?? new ApiRecommendations();
    }

    /// <summary>
    /// Asks Jellyseerr to request a film.
    /// </summary>
    /// <param name="tmdbId">TMDB id of the film.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The HTTP status Jellyseerr answered with.</returns>
    public async Task<int> RequestMovieAsync(int tmdbId, CancellationToken cancellationToken)
    {
        using var request = new HttpRequestMessage(
            HttpMethod.Post,
            BuildUrl(Config.JellyseerrUrl, "api/v1/request"));
        request.Headers.Add("X-Api-Key", Config.JellyseerrApiKey);
        request.Content = JsonContent.Create(new { mediaType = "movie", mediaId = tmdbId });

        var client = _httpClientFactory.CreateClient(NamedClient.Default);
        using var response = await client.SendAsync(request, cancellationToken).ConfigureAwait(false);
        return (int)response.StatusCode;
    }

    private static Configuration.PluginConfiguration Config =>
        Plugin.Instance?.Configuration ?? new Configuration.PluginConfiguration();

    private static Uri BuildUrl(string baseUrl, string path)
    {
        return new Uri(baseUrl.TrimEnd('/') + "/" + path);
    }
}
