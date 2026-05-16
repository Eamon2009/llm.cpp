#!/usr/bin/env node

const { createServer } = require("node:http");
const { createReadStream, existsSync, mkdirSync, statSync } = require("node:fs");
const { extname, join, resolve } = require("node:path");
const { spawn } = require("node:child_process");
const { platform } = require("node:os");

const packageRoot = resolve(__dirname, "..");
const userRoot = process.cwd();
const isWindows = platform() === "win32";
const python = isWindows ? "python" : "python3";
const cppBinary = join(userRoot, ".quadtrix", "bin", isWindows ? "quadtrix.exe" : "quadtrix");

const mimeTypes = {
  ".css": "text/css",
  ".html": "text/html",
  ".ico": "image/x-icon",
  ".js": "text/javascript",
  ".json": "application/json",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".webmanifest": "application/manifest+json",
};

function usage() {
  console.log(`
Quadtrix CLI

Usage:
  quadtrix chat [--api-port 3001] [--web-port 5173] [--no-open]
  quadtrix train --backend cpp [--data data/input.txt]
  quadtrix train --backend python
  quadtrix setup

Commands:
  chat      Start the FastAPI backend, serve the built frontend, and open chat.
  train     Train locally with either the C++ or Python implementation.
  setup     Install Python backend/engine dependencies with pip.
`);
}

function argValue(args, name, fallback) {
  const index = args.indexOf(name);
  if (index === -1 || index + 1 >= args.length) {
    return fallback;
  }
  return args[index + 1];
}

function hasArg(args, name) {
  return args.includes(name);
}

function run(command, args, options = {}) {
  const child = spawn(command, args, {
    cwd: options.cwd || userRoot,
    env: { ...process.env, ...(options.env || {}) },
    stdio: options.stdio || "inherit",
    shell: false,
  });

  child.on("error", (error) => {
    console.error(`Failed to start ${command}: ${error.message}`);
  });

  return child;
}

function openBrowser(url) {
  if (isWindows) {
    spawn("cmd", ["/c", "start", "", url], { detached: true, stdio: "ignore" }).unref();
    return;
  }

  if (platform() === "darwin") {
    spawn("open", [url], { detached: true, stdio: "ignore" }).unref();
    return;
  }

  spawn("xdg-open", [url], { detached: true, stdio: "ignore" }).unref();
}

function serveStatic(directory, port) {
  if (!existsSync(directory)) {
    console.error(`Frontend build not found: ${directory}`);
    console.error("Run `npm run build:frontend` before packing or publishing.");
    process.exit(1);
  }

  const server = createServer((request, response) => {
    const rawPath = decodeURIComponent((request.url || "/").split("?")[0]);
    const safePath = rawPath.replace(/^\/+/, "");
    let filePath = resolve(directory, safePath || "index.html");

    if (!filePath.startsWith(resolve(directory))) {
      response.writeHead(403);
      response.end("Forbidden");
      return;
    }

    if (!existsSync(filePath) || statSync(filePath).isDirectory()) {
      filePath = join(directory, "index.html");
    }

    response.writeHead(200, {
      "Content-Type": mimeTypes[extname(filePath)] || "application/octet-stream",
    });
    createReadStream(filePath).pipe(response);
  });

  server.listen(port, () => {
    console.log(`Frontend: http://localhost:${port}`);
  });

  return server;
}

function startChat(args) {
  const apiPort = argValue(args, "--api-port", "3001");
  const webPort = argValue(args, "--web-port", "5173");
  const frontendDist = join(packageRoot, "frontend", "dist");
  const backendDir = join(packageRoot, "backend");
  const url = `http://localhost:${webPort}`;

  const api = run(python, ["-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", apiPort], {
    cwd: backendDir,
    env: {
      API_PORT: apiPort,
      CORS_ORIGINS: url,
    },
  });
  const web = serveStatic(frontendDist, Number(webPort));

  console.log("Starting Quadtrix chat...");
  console.log(`Backend:  http://localhost:${apiPort}`);

  if (!hasArg(args, "--no-open")) {
    setTimeout(() => openBrowser(url), 1200);
  }

  const stop = () => {
    web.close();
    api.kill();
    process.exit(0);
  };

  process.on("SIGINT", stop);
  process.on("SIGTERM", stop);
}

function setup() {
  const requirements = join(packageRoot, "backend", "requirements.txt");
  const child = run(python, ["-m", "pip", "install", "-r", requirements]);
  child.on("exit", (code) => process.exit(code || 0));
}

function compileCpp() {
  mkdirSync(join(userRoot, ".quadtrix", "bin"), { recursive: true });
  const child = run("g++", [
    "-std=c++17",
    "-O2",
    "-I.",
    "-Iinclude",
    "-o",
    cppBinary,
    "main.cpp",
  ], { cwd: packageRoot });

  return new Promise((resolvePromise) => {
    child.on("exit", (code) => resolvePromise(code || 0));
  });
}

function resolveTrainingData(args) {
  const requested = argValue(args, "--data", join(userRoot, "data", "input.txt"));
  const data = resolve(userRoot, requested);

  if (!existsSync(data)) {
    console.error(`Training data not found: ${data}`);
    console.error("Pass a text file with `--data ./path/to/input.txt`.");
    process.exit(1);
  }

  return data;
}

async function train(args) {
  const backend = argValue(args, "--backend", "cpp");

  if (backend === "cpp") {
    const data = resolveTrainingData(args);
    const code = await compileCpp();
    if (code !== 0) {
      process.exit(code);
    }
    run(cppBinary, [data]).on("exit", (exitCode) => process.exit(exitCode || 0));
    return;
  }

  if (backend === "python" || backend === "py") {
    const data = resolveTrainingData(args);
    const script = join(packageRoot, "engine", "main.py");
    run(python, [script], {
      cwd: join(packageRoot, "engine"),
      env: { QUADTRIX_TRAIN_DATA: data },
    }).on("exit", (exitCode) => {
      process.exit(exitCode || 0);
    });
    return;
  }

  console.error(`Unknown backend: ${backend}`);
  console.error("Use `--backend cpp` or `--backend python`.");
  process.exit(1);
}

async function main() {
  const [command, ...args] = process.argv.slice(2);

  if (!command || command === "--help" || command === "-h") {
    usage();
    return;
  }

  if (command === "chat") {
    startChat(args);
    return;
  }

  if (command === "setup") {
    setup();
    return;
  }

  if (command === "train") {
    await train(args);
    return;
  }

  console.error(`Unknown command: ${command}`);
  usage();
  process.exit(1);
}

main();
