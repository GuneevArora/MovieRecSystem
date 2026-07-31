using Jellyfin.Plugin.MovieRec.Configuration;
using MediaBrowser.Common.Configuration;
using MediaBrowser.Common.Plugins;
using MediaBrowser.Model.Plugins;
using MediaBrowser.Model.Serialization;

namespace Jellyfin.Plugin.MovieRec;

/// <summary>
/// Recommends films from a user's watch history using the MovieRecSystem API.
/// </summary>
public class Plugin : BasePlugin<PluginConfiguration>, IHasWebPages
{
    public Plugin(IApplicationPaths applicationPaths, IXmlSerializer xmlSerializer)
        : base(applicationPaths, xmlSerializer)
    {
        Instance = this;
    }

    public static Plugin? Instance { get; private set; }

    public override string Name => "MovieRec";

    public override Guid Id => Guid.Parse("6f2d1c7e-3b0a-4e58-9c41-7a5d2e9b8f13");

    public override string Description =>
        "Personal film recommendations from a sequential two-tower model, with Jellyseerr requests.";

    public IEnumerable<PluginPageInfo> GetPages()
    {
        return
        [
            new PluginPageInfo
            {
                Name = Name,
                EmbeddedResourcePath = $"{GetType().Namespace}.Configuration.configPage.html",
            },
        ];
    }
}
