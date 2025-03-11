import { createRouter, createWebHashHistory } from 'vue-router';
import generatedRoutes from 'virtual:generated-pages';

const router = createRouter({
  history: createWebHashHistory(), // 修改为 Hash 模式，避免 GitHub Pages 404
  routes: generatedRoutes,
});

export default router;
