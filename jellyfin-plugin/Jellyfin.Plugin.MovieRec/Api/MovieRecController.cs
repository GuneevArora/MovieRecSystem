using System.Net.Mime;
using MediaBrowser.Controller.Library;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;

namespace Jellyfin.Plugin.MovieRec.Api;

/// <summary>
/// HTTP endpoints the plugin adds to the Jellyfin server.
/// </summary>
[ApiController]
[Authorize]
[Route("MovieRec")]
[Produces(MediaTypeNames.Application.Json)]
public class MovieRecController : ControllerBase
{
    private const string UserIdClaim = "Jellyfin-UserId";
    private const string AdministratorRole = "Administrator";

    private readonly IUserManager _userManager;
    private readonly RecommendationService _recommendations;
    private readonly RecommenderClient _client;

    public MovieRecController(
        IUserManager userManager,
        RecommendationService recommendations,
        RecommenderClient client)
    {
        _userManager = userManager;
        _recommendations = recommendations;
        _client = client;
    }

    /// <summary>
    /// Recommendations for the signed-in user. Administrators may ask for another user.
    /// </summary>
    /// <param name="userId">Optional user to look up; administrators only.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Films in the library and films missing from it.</returns>
    [HttpGet("Recommendations")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    public async Task<ActionResult<UserRecommendations>> GetRecommendations(
        [FromQuery] Guid? userId,
        CancellationToken cancellationToken)
    {
        if (!Guid.TryParse(User.FindFirst(UserIdClaim)?.Value, out var callerId))
        {
            return Unauthorized();
        }

        var targetId = userId ?? callerId;
        if (targetId != callerId && !User.IsInRole(AdministratorRole))
        {
            return Forbid();
        }

        var user = _userManager.GetUserById(targetId);
        if (user is null)
        {
            return NotFound();
        }

        var config = Plugin.Instance!.Configuration;
        try
        {
            return await _recommendations
                .GetAsync(user, config.PlaylistSize, config.DiscoverCount, cancellationToken)
                .ConfigureAwait(false);
        }
        catch (HttpRequestException)
        {
            return StatusCode(StatusCodes.Status502BadGateway, "The recommender API is not reachable.");
        }
    }

    /// <summary>
    /// Requests a film through Jellyseerr. The plugin uses Jellyseerr's API key,
    /// which acts with full rights there, so this is limited to administrators.
    /// </summary>
    /// <param name="tmdbId">TMDB id of the film.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>No content on success.</returns>
    [HttpPost("Request/{tmdbId:int}")]
    [Authorize(Roles = AdministratorRole)]
    [ProducesResponseType(StatusCodes.Status204NoContent)]
    public async Task<ActionResult> RequestMovie([FromRoute] int tmdbId, CancellationToken cancellationToken)
    {
        var config = Plugin.Instance!.Configuration;
        if (string.IsNullOrWhiteSpace(config.JellyseerrUrl) || string.IsNullOrWhiteSpace(config.JellyseerrApiKey))
        {
            return BadRequest("Jellyseerr is not configured.");
        }

        try
        {
            var status = await _client.RequestMovieAsync(tmdbId, cancellationToken).ConfigureAwait(false);
            return status is >= 200 and < 300
                ? NoContent()
                : StatusCode(StatusCodes.Status502BadGateway, $"Jellyseerr answered {status}.");
        }
        catch (HttpRequestException)
        {
            return StatusCode(StatusCodes.Status502BadGateway, "Jellyseerr is not reachable.");
        }
    }
}
