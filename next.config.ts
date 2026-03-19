import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Exclude server-only ONNX and sharp packages from the browser bundle,
  // enable top-level await for WASM modules, and serve .wasm as assets.
  webpack: (config) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      sharp$: false,
      "onnxruntime-node$": false,
    };

    // Required for @xenova/transformers WASM async initialisation
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
  // Required for SharedArrayBuffer used by ONNX multi-thread backend
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
          // "credentialless" still enables crossOriginIsolated (SharedArrayBuffer)
          // but also allows CDN imports in module workers without CORP headers.
          { key: "Cross-Origin-Embedder-Policy", value: "credentialless" },
        ],
      },
    ];
  },
};

export default nextConfig;
