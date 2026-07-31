using MediaBrowser.Controller.Library;
using MediaBrowser.Controller.Playlists;
using MediaBrowser.Model.Playlists;
using MediaBrowser.Model.Tasks;
using Microsoft.Extensions.Logging;

namespace Jellyfin.Plugin.MovieRec.ScheduledTasks;

/// <summary>
/// Keeps one private playlist per user filled with the films in the library
/// the model ranks highest for them. A playlist shows up in every Jellyfin
/// client, so no client changes are needed.
/// </summary>
public class RefreshPlaylistsTask : IScheduledTask
{
    private readonly IUserManager _userManager;
    private readonly IPlaylistManager _playlistManager;
    private readonly RecommendationService _recommendations;
    private readonly ILogger<RefreshPlaylistsTask> _logger;

    public RefreshPlaylistsTask(
        IUserManager userManager,
        IPlaylistManager playlistManager,
        RecommendationService recommendations,
        ILogger<RefreshPlaylistsTask> logger)
    {
        _userManager = userManager;
        _playlistManager = playlistManager;
        _recommendations = recommendations;
        _logger = logger;
    }

    public string Name => "Refresh MovieRec playlists";

    public string Key => "MovieRecRefreshPlaylists";

    public string Description => "Rebuilds each user's recommendation playlist from their watch history.";

    public string Category => "MovieRec";

    public IEnumerable<TaskTriggerInfo> GetDefaultTriggers()
    {
        return
        [
            new TaskTriggerInfo
            {
                Type = TaskTriggerInfoType.DailyTrigger,
                TimeOfDayTicks = TimeSpan.FromHours(4).Ticks,
            },
        ];
    }

    public async Task ExecuteAsync(IProgress<double> progress, CancellationToken cancellationToken)
    {
        var config = Plugin.Instance?.Configuration;
        if (config is null || string.IsNullOrWhiteSpace(config.PlaylistName) || config.PlaylistSize <= 0)
        {
            return;
        }

        var users = _userManager.Users.ToList();
        for (var i = 0; i < users.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var user = users[i];
            try
            {
                var result = await _recommendations
                    .GetAsync(user, config.PlaylistSize, 0, cancellationToken)
                    .ConfigureAwait(false);

                // No history means the API can only offer popularity, which is not personal.
                if (result.HistoryCount > 0 && result.InLibrary.Count > 0)
                {
                    var itemIds = result.InLibrary.Select(pick => pick.ItemId).ToArray();
                    await ReplacePlaylistAsync(user.Id, config.PlaylistName, itemIds).ConfigureAwait(false);
                    _logger.LogInformation(
                        "MovieRec: {Count} films for {User} from {History} watched",
                        itemIds.Length,
                        user.Username,
                        result.HistoryCount);
                }
            }
            catch (HttpRequestException ex)
            {
                _logger.LogError(ex, "MovieRec: recommender API not reachable at {Url}", config.ApiUrl);
                throw;
            }

            progress.Report(100.0 * (i + 1) / users.Count);
        }
    }

    private async Task ReplacePlaylistAsync(Guid userId, string name, Guid[] itemIds)
    {
        var existing = _playlistManager
            .GetPlaylists(userId)
            .FirstOrDefault(playlist => playlist.OwnerUserId.Equals(userId)
                && string.Equals(playlist.Name, name, StringComparison.Ordinal));

        if (existing is null)
        {
            await _playlistManager.CreatePlaylist(new PlaylistCreationRequest
            {
                Name = name,
                ItemIdList = itemIds,
                UserId = userId,
                MediaType = Data.Enums.MediaType.Video,
                Public = false,
            }).ConfigureAwait(false);
            return;
        }

        var current = existing.LinkedChildren
            .Where(child => child.ItemId.HasValue)
            .Select(child => child.ItemId!.Value)
            .ToArray();
        if (current.SequenceEqual(itemIds))
        {
            return;
        }

        if (current.Length > 0)
        {
            await _playlistManager
                .RemoveItemFromPlaylistAsync(
                    existing.Id.ToString("N"),
                    current.Select(id => id.ToString("N")))
                .ConfigureAwait(false);
        }

        await _playlistManager.AddItemToPlaylistAsync(existing.Id, itemIds, userId).ConfigureAwait(false);
    }
}
