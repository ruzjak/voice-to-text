import type { NextConfig } from "next";

// COOP/COEP headers are set in vercel.json for production.
// The webpack config below handles WASM bundling for the ONNX backend.
const nextConfig: NextConfig = {
  webpack: (config) => {
    // Exclude server-only packages from the browser bundle
    config.resolve.alias = {
      ...config.resolve.alias,
      sharp$: false,
      "onnxruntime-node$": false,
    };

    // Required for WASM async initialisation
    config.experiments = {
      ...config.experiments,
      topLevelAwait: true,
    };

    // Emit .wasm files as assets so they can be fetched at runtime
    config.module.rules.push({
      test: /\.wasm$/,
      type: "asset/resource",
    });

    return config;
  },
};

export default nextConfig;
