#!/usr/bin/env python3
"""Static file server with HTTP Range support (required for PMTiles).

Usage: python3 server.py [port] [directory]
"""
import argparse
import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

MIME = {
    ".html": "text/html",
    ".js": "text/javascript",
    ".pmtiles": "application/octet-stream",
    ".tif": "image/tiff",
    ".webp": "image/webp",
}


class RangeFile:
    """File-like object serving a byte range of a larger file."""

    def __init__(self, fd, start, length):
        self.fd = fd
        self.pos = start
        self.end = start + length

    def read(self, n=-1):
        if self.pos >= self.end:
            return b""
        if n is None or n < 0:
            n = self.end - self.pos
        n = min(n, self.end - self.pos)
        os.lseek(self.fd, self.pos, os.SEEK_SET)
        data = os.read(self.fd, n)
        self.pos += len(data)
        return data

    def close(self):
        os.close(self.fd)


class RangeHTTPRequestHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        ".html": "text/html",
        ".js": "text/javascript",
        "": "application/octet-stream",
    }

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def guess_type(self, path):
        base, ext = os.path.splitext(path)
        return MIME.get(ext.lower(), super().guess_type(path))

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            path = os.path.join(path, "index.html")
        if not os.path.isfile(path):
            self.send_error(404, "File not found")
            return None

        ctype = self.guess_type(path)
        size = os.path.getsize(path)
        fd = os.open(path, os.O_RDONLY)

        range_header = self.headers.get("Range")
        m = re.match(r"bytes=(\d*)-(\d*)", range_header.strip()) if range_header else None
        if m:
            start = int(m.group(1)) if m.group(1) else 0
            end = int(m.group(2)) if m.group(2) else size - 1
            end = min(end, size - 1)
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-type", ctype)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.send_header("Last-Modified", self.date_time_string())
            self.end_headers()
            return RangeFile(fd, start, length)

        self.send_response(200)
        self.send_header("Content-type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Last-Modified", self.date_time_string())
        self.end_headers()
        return RangeFile(fd, 0, size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("port", nargs="?", type=int, default=8000)
    parser.add_argument("directory", nargs="?", default=".")
    args = parser.parse_args()
    handler = partial(RangeHTTPRequestHandler, directory=os.path.abspath(args.directory))
    server = ThreadingHTTPServer(("0.0.0.0", args.port), handler)
    print(f"Serving {os.path.abspath(args.directory)} on http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
