import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Plain static files: the model runs in the browser, so any static host works.
  output: "export",
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
