<template>
  <div class="blog-detail">
    <el-button @click="goBack" class="back-button">返回</el-button>

    <!-- 根据文件类型渲染不同内容 -->
    <div v-if="isMarkdown" v-html="renderedMarkdown" class="markdown-content markdown-body"></div>
    <div v-if="isNotebook" v-html="renderedNotebook" class="notebook-content"></div>
    <div v-if="!isMarkdown && !isNotebook" class="unsupported-file">不支持的文件格式</div>
  </div>
</template>

<script lang="ts" setup>
import { ref, onMounted, nextTick } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import MarkdownIt from 'markdown-it';
import markdownItMathjax3 from 'markdown-it-mathjax3';
import hljs from 'highlight.js';
import 'highlight.js/styles/github.css'; // 代码高亮样式

// Vue Router 相关
const route = useRoute();
const router = useRouter();

// 状态变量
const renderedMarkdown = ref('');
const renderedNotebook = ref('');
const isMarkdown = ref(false);
const isNotebook = ref(false);

// Markdown 解析器
const mdParser = new MarkdownIt({
  html: true,
  highlight: (code, lang) => {
    if (lang && hljs.getLanguage(lang)) {
      try {
        return `<pre class="hljs"><code>${hljs.highlight(code, { language: lang }).value}</code></pre>`;
      } catch (__) {}
    }
    return `<pre class="hljs"><code>${mdParser.utils.escapeHtml(code)}</code></pre>`;
  },
}).use(markdownItMathjax3);

// 组件挂载时加载文件
onMounted(async () => {
  const filePath = route.query.path as string;
  if (!filePath) {
    return;
  }

  isMarkdown.value = filePath.endsWith('.md');
  isNotebook.value = filePath.endsWith('.ipynb');

  if (isMarkdown.value) {
    await loadMarkdown(filePath);
  } else if (isNotebook.value) {
    await loadNotebook(filePath);
  }

  // Vue 渲染完成后再执行代码高亮
  nextTick(() => {
    document.querySelectorAll('pre code').forEach((el) => {
      hljs.highlightElement(el as HTMLElement);
    });
  });
});

// 加载 Markdown 文件
const loadMarkdown = async (path: string) => {
  try {
    const response = await fetch(path);
    if (!response.ok) throw new Error('Markdown 文件加载失败');
    const text = await response.text();
    renderedMarkdown.value = mdParser.render(text);
  } catch (error) {
    console.error('Markdown 加载失败:', error);
    renderedMarkdown.value = '<p>无法加载 Markdown 文件</p>';
  }
};

// 解析 Notebook 输出
const parseOutputs = (outputs: any[]) => {
  let outputHtml = '';
  outputs.forEach((output) => {
    if (output.data) {
      if (output.data['text/html']) {
        outputHtml += output.data['text/html'].join('');
      } else if (output.data['image/png']) {
        outputHtml += `<img src="data:image/png;base64,${output.data['image/png']}" />`;
      } else if (output.data['text/plain']) {
        outputHtml += `<pre class="hljs"><code>${hljs.highlight(output.data['text/plain'].join(''), { language: 'plaintext' }).value}</code></pre>`;
      }
    }
  });
  return outputHtml;
};

// 加载 Notebook 文件
const loadNotebook = async (path: string) => {
  try {
    const response = await fetch(path);
    if (!response.ok) throw new Error('Notebook 文件加载失败');
    const notebookData = await response.json();

    let htmlContent = '';
    notebookData.cells.forEach((cell: any) => {
      if (cell.cell_type === 'markdown') {
        let markdownContent = cell.source.join('');

        if (cell.attachments) {
          Object.keys(cell.attachments).forEach((filename) => {
            const attachment = cell.attachments[filename];
            const imageMimeType = Object.keys(attachment)[0];
            const base64Data = attachment[imageMimeType];

            markdownContent = markdownContent.replace(
              new RegExp(`!\\[.*\\]\\(attachment:${filename}\\)`, 'g'),
              `![${filename}](data:${imageMimeType};base64,${base64Data})`
            );
          });
        }
        htmlContent += mdParser.render(markdownContent);
      } else if (cell.cell_type === 'code') {
        const lang = cell.metadata?.language || 'python';
        htmlContent += `<pre class="hljs"><code>${hljs.highlight(cell.source.join(''), { language: lang }).value}</code></pre>`;
        if (cell.outputs && cell.outputs.length > 0) {
          htmlContent += parseOutputs(cell.outputs);
        }
      }
    });

    renderedNotebook.value = htmlContent;
  } catch (error) {
    console.error('Notebook 解析失败:', error);
    renderedNotebook.value = '<p>无法加载 Jupyter Notebook</p>';
  }
};

// 返回按钮逻辑
const goBack = () => {
  router.push({ path: '/blogs' });
};
</script>

<style>
.blog-detail {
  width: 50%;
  margin: 0 auto;
  padding: 20px;
  text-align: left;
}

.back-button {
  margin-bottom: 15px;
}

.markdown-content,
.notebook-content {
  border: 1px solid #ddd;
  padding: 20px;
  border-radius: 8px;
  text-align: left;
}

.markdown-content img,
.notebook-content img {
  max-width: 100% !important;
  height: auto !important;
  display: block;
  margin: 0 auto;
}

/* 代码块样式 */
pre {
  border: 1px solid #ddd;
  padding: 10px;
  overflow-x: auto;
  border-radius: 5px;
  margin-bottom: 10px;
  white-space: pre-wrap;
}
</style>
