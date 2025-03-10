<template>
  <div class="blog-detail">
    <el-button @click="goBack" class="back-button">返回</el-button>
    <div v-html="renderedContent" class="markdown-content markdown-body"></div>
  </div>
</template>

<script lang="ts" setup>
import { ref, onMounted } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import MarkdownIt from 'markdown-it';
import markdownItMathjax3 from 'markdown-it-mathjax3';

const route = useRoute();
const router = useRouter();
const renderedContent = ref('');
const mdParser = new MarkdownIt({ html: true }).use(markdownItMathjax3);

onMounted(async () => {
  const filePath = route.query.path as string;
  if (filePath) {
    try {
      const response = await fetch(filePath);
      const text = await response.text();

      if (filePath.endsWith('.md')) {
        // 解析 Markdown
        renderedContent.value = mdParser.render(text);
      } else if (filePath.endsWith('.ipynb')) {
        // 解析 Jupyter Notebook
        parseNotebook(JSON.parse(text));
      }
    } catch (error) {
      renderedContent.value = '<p>无法加载博客内容</p>';
    }
  }
});

/**
 * 解析 Jupyter Notebook 文件
 */
const parseNotebook = (notebook: any) => {
  let content = '';

  notebook.cells.forEach((cell: any) => {
    if (cell.cell_type === 'markdown') {
      // 解析 Markdown 单元格
      content += mdParser.render(cell.source.join(''));
    } else if (cell.cell_type === 'code') {
      // 解析代码单元格
      const codeContent = cell.source.join('');
      content += `<pre><code>${escapeHtml(codeContent)}</code></pre>`;
    }
  });

  renderedContent.value = content;
};

/**
 * 转义 HTML（避免代码单元格中的特殊字符导致 HTML 解析错误）
 */
const escapeHtml = (str: string) => {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
};

const goBack = () => {
  router.push({ name: '/blogs' });
};
</script>

<style>
.blog-detail {
  width: 50%;
  margin: 0 auto;
  padding: 20px;
}

.back-button {
  margin-bottom: 15px;
}

.markdown-content {
  border: 1px solid #ddd;
  padding: 20px;
  border-radius: 8px;
}

.markdown-body img {
  max-width: 100% !important;
  height: auto !important;
  display: block;
  margin: 0 auto;
}

pre {
  background: #282c34;
  color: #fff;
  padding: 10px;
  border-radius: 8px;
  overflow-x: auto;
}

code {
  font-family: monospace;
}
</style>
