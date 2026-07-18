import type { NextConfig } from "next";

const nextConfig: NextConfig = {
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
