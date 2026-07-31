using System.Globalization;
using Jellyfin.Data.Enums;
using Jellyfin.Database.Implementations.Entities;
using MediaBrowser.Controller.Entities;
using MediaBrowser.Controller.Library;
using MediaBrowser.Model.Entities;

namespace Jellyfin.Plugin.MovieRec;

/// <summary>A recommended film that is already in the library.</summary>
public record LibraryPick(Guid ItemId, string Name, int? Year, int TmdbId, double? Score);

/// <summary>Recommendations for one user.</summary>
public record UserRecommendations(
    int HistoryCount,
    string Strategy,
    IReadOnlyList<LibraryPick> InLibrary,
    IReadOnlyList<ApiMovie> Missing);

/// <summary>
/// Turns a Jellyfin user's watch history into recommendations.
/// </summary>
public class RecommendationService
{
    // The API accepts at most this many history items; only the last 50 reach the model anyway.
    private const int MaxHistory = 500;
    private const int MaxLibrary = 50_000;

    private readonly ILibraryManager _libraryManager;
    private readonly IUserDataManager _userDataManager;
    private readonly RecommenderClient _client;

    public RecommendationService(
        ILibraryManager libraryManager,
        IUserDataManager userDataManager,
        RecommenderClient client)
    {
        _libraryManager = libraryManager;
        _userDataManager = userDataManager;
        _client = client;
    }

    /// <summary>
    /// Builds both lists for a user: the best unwatched films they can already
    /// play, and the best films the library does not have.
    /// </summary>
    /// <param name="user">The Jellyfin user.</param>
    /// <param name="librarySize">How many library films to return.</param>
    /// <param name="discoverSize">How many missing films to return; 0 skips the call.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The recommendations.</returns>
    public async Task<UserRecommendations> GetAsync(
        User user,
        int librarySize,
        int discoverSize,
        CancellationToken cancellationToken)
    {
        var includeFavorites = Plugin.Instance?.Configuration.IncludeFavorites ?? true;

        // Scoped to the user, so libraries they cannot see are never recommended.
        var movies = _libraryManager.GetItemList(new InternalItemsQuery(user)
        {
            IncludeItemTypes = [BaseItemKind.Movie],
            IsVirtualItem = false,
            Recursive = true,
        });

        var liked = new List<(int TmdbId, DateTime When)>();
        var unwatched = new Dictionary<int, BaseItem>();
        var owned = new HashSet<int>();
        foreach (var movie in movies)
        {
            if (!TryGetTmdbId(movie, out var tmdbId))
            {
                continue;
            }

            owned.Add(tmdbId);
            var data = _userDataManager.GetUserData(user, movie);
            var played = data is not null && data.Played;
            if (played || (includeFavorites && data is not null && data.IsFavorite))
            {
                liked.Add((tmdbId, data!.LastPlayedDate ?? DateTime.MinValue));
            }

            if (!played)
            {
                unwatched.TryAdd(tmdbId, movie);
            }
        }

        // The model reads the history oldest first.
        var history = liked
            .OrderBy(entry => entry.When)
            .Select(entry => entry.TmdbId)
            .Distinct()
            .TakeLast(MaxHistory)
            .ToList();

        var inLibrary = new List<LibraryPick>();
        var strategy = "none";
        if (librarySize > 0 && unwatched.Count > 0)
        {
            var response = await _client.RecommendAsync(
                history,
                librarySize,
                unwatched.Keys.Take(MaxLibrary).ToList(),
                null,
                cancellationToken).ConfigureAwait(false);
            strategy = response.Strategy;
            foreach (var pick in response.Items)
            {
                if (pick.TmdbId is int id && unwatched.TryGetValue(id, out var item))
                {
                    inLibrary.Add(new LibraryPick(item.Id, item.Name, item.ProductionYear, id, pick.Score));
                }
            }
        }

        IReadOnlyList<ApiMovie> missing = [];
        if (discoverSize > 0)
        {
            var response = await _client.RecommendAsync(
                history,
                discoverSize,
                null,
                owned.Take(MaxLibrary).ToList(),
                cancellationToken).ConfigureAwait(false);
            strategy = response.Strategy;
            missing = response.Items.Where(movie => movie.TmdbId is not null).ToList();
        }

        return new UserRecommendations(history.Count, strategy, inLibrary, missing);
    }

    private static bool TryGetTmdbId(BaseItem item, out int tmdbId)
    {
        tmdbId = 0;
        return item.TryGetProviderId(MetadataProvider.Tmdb, out var raw)
            && int.TryParse(raw, NumberStyles.None, CultureInfo.InvariantCulture, out tmdbId);
    }
}
