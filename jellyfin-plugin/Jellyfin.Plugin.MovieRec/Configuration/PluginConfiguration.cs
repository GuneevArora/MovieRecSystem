using MediaBrowser.Model.Plugins;

namespace Jellyfin.Plugin.MovieRec.Configuration;

/// <summary>
/// Settings shown on the plugin's page in the Jellyfin dashboard.
/// </summary>
public class PluginConfiguration : BasePluginConfiguration
{
    /// <summary>Gets or sets the base URL of the recommender API.</summary>
    public string ApiUrl { get; set; } = "http://localhost:8000";

    /// <summary>Gets or sets the name of the playlist kept for each user.</summary>
    public string PlaylistName { get; set; } = "Recommended for you";

    /// <summary>Gets or sets how many library films go into the playlist.</summary>
    public int PlaylistSize { get; set; } = 20;

    /// <summary>Gets or sets how many films outside the library to suggest.</summary>
    public int DiscoverCount { get; set; } = 12;

    /// <summary>Gets or sets a value indicating whether favourites count as liked even if unplayed.</summary>
    public bool IncludeFavorites { get; set; } = true;

    /// <summary>Gets or sets the base URL of Jellyseerr. Empty turns requests off.</summary>
    public string JellyseerrUrl { get; set; } = string.Empty;

    /// <summary>Gets or sets the Jellyseerr API key.</summary>
    public string JellyseerrApiKey { get; set; } = string.Empty;
}
