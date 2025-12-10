import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
 plugins: [react()],
 css: {
  preprocessorOptions: {
   scss: {
    silenceDeprecations: ["legacy-js-api"],
   },
  },
 },
 server: {
  port: 3000,
  open: true,
  proxy: {
   "/api": {
    target: "http://127.0.0.1:5000",
    changeOrigin: true,
    secure: false,
   },
   // RAG Flask on 8000
   "/rag": {
    target: "http://127.0.0.1:8000",
    changeOrigin: true,
    secure: false,
   },
  },
 },
});
