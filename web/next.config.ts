import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Plain static files: the model runs in the browser, so any static host works.
  output: "export",
  // Set NEXT_PUBLIC_BASE_PATH (for example /movierecdemo) to host under a sub-path.
  basePath: process.env.NEXT_PUBLIC_BASE_PATH || undefined,
  images: {
    // Posters come straight from TMDB's image CDN, already sized.
    unoptimized: true,
    remotePatterns: [
      {
        protocol: "https",
        hostname: "image.tmdb.org",
      },
    ],
  },
};

export default nextConfig;
