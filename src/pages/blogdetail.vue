<template>
  <div class="blog-detail">
    <el-button @click="goBack" class="back-button">返回</el-button>
    <!-- 根据文件类型显示不同内容 -->
    <div v-if="isMarkdown" v-html="renderedContent" class="markdown-content markdown-body"></div>
    <div v-else v-html="renderedNotebook" class="notebook-content"></div>
  </div>
</template>

<script lang="ts" setup>
import { ref, onMounted } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import MarkdownIt from 'markdown-it';
import markdownItMathjax3 from 'markdown-it-mathjax3';

// 路由及状态初始化
const route = useRoute();
const router = useRouter();
const renderedContent = ref('');   // 用于 Markdown 文件
const renderedNotebook = ref('');  // 用于 ipynb 文件
const isMarkdown = ref(true);

// 配置 Markdown 解析器，支持 HTML 与 MathJax
const mdParser = new MarkdownIt({ html: true }).use(markdownItMathjax3);

// 页面加载时，根据文件后缀决定如何解析
onMounted(async () => {
  const filePath = route.query.path as string;
  console.log('加载的文件路径:', filePath);

  if (!filePath) {
    renderedContent.value = '<p>未提供有效的文件路径</p>';
    return;
  }

  if (filePath.endsWith('.md')) {
    isMarkdown.value = true;
    await loadMarkdown(filePath);
  } else if (filePath.endsWith('.ipynb')) {
    isMarkdown.value = false;
    await loadNotebook(filePath);
  } else {
    renderedContent.value = '<p>不支持的文件格式</p>';
  }
});

// 加载 Markdown 文件
const loadMarkdown = async (path: string) => {
  try {
    const response = await fetch(path);
    if (!response.ok) throw new Error('Markdown 文件加载失败');
    const text = await response.text();
    renderedContent.value = mdParser.render(text);
  } catch (error) {
    console.error('Markdown 加载失败:', error);
    renderedContent.value = '<p>无法加载 Markdown 文件</p>';
  }
};

// 处理 Notebook cell 输出：支持 text/html、image/png、text/plain
const parseOutputs = (outputs: any[]) => {
  let outputHtml = '';
  outputs.forEach((output) => {
    if (output.data) {
      if (output.data['text/html']) {
        outputHtml += output.data['text/html'].join('');
      } else if (output.data['image/png']) {
        outputHtml += `<img src="data:image/png;base64,${output.data['image/png']}" />`;
      } else if (output.data['text/plain']) {
        outputHtml += `<pre>${output.data['text/plain'].join('')}</pre>`;
      }
    }
  });
  return outputHtml;
};

// 转义 HTML，防止代码单元中出现特殊字符引起解析问题
const escapeHtml = (str: string) => {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
};

// 加载并解析 .ipynb 文件
// const loadNotebook = async (path: string) => {
//   try {
//     const response = await fetch(path);
//     if (!response.ok) throw new Error('Notebook 文件加载失败');
//     const notebookData = await response.json();

//     let htmlContent = '';

//     // 遍历 notebook 中的每个 cell
//     notebookData.cells.forEach((cell: any) => {
//       if (cell.cell_type === 'markdown') {
//         // Markdown cell 使用 markdown-it 渲染
//         htmlContent += mdParser.render(cell.source.join(''));
//       } else if (cell.cell_type === 'code') {
//         // 显示代码单元的原始代码
//         htmlContent += `<pre class="code-cell"><code>${escapeHtml(cell.source.join(''))}</code></pre>`;
//         // 如果代码单元有输出，解析并显示输出
//         if (cell.outputs && cell.outputs.length > 0) {
//           htmlContent += parseOutputs(cell.outputs);
//         }
//       }
//     });

//     renderedNotebook.value = htmlContent;
//   } catch (error) {
//     console.error('Notebook 解析失败:', error);
//     renderedNotebook.value = '<p>无法加载 Jupyter Notebook</p>';
//   }
// };

const loadNotebook = async (path: string) => {
  try {
    const response = await fetch(path);
    if (!response.ok) throw new Error('Notebook 文件加载失败');
    const notebookData = await response.json();

    let htmlContent = '';

    // 遍历 notebook 中的每个 cell
    notebookData.cells.forEach((cell: any) => {
      if (cell.cell_type === 'markdown') {
        let markdownContent = cell.source.join('');

        // 处理 Markdown cell 的附件
        if (cell.attachments) {
          Object.keys(cell.attachments).forEach((filename) => {
            const attachment = cell.attachments[filename];
            const imageMimeType = Object.keys(attachment)[0]; // 获取附件的 MIME 类型
            const base64Data = attachment[imageMimeType]; // 获取 base64 编码的图片数据

            // 替换 Markdown 中的附件引用为 base64 图片
            markdownContent = markdownContent.replace(
              new RegExp(`!\\[.*\\]\\(attachment:${filename}\\)`, 'g'),
              `![${filename}](data:${imageMimeType};base64,${base64Data})`
            );
          });
        }

        // 使用 markdown-it 渲染 Markdown 内容
        htmlContent += mdParser.render(markdownContent);
      } else if (cell.cell_type === 'code') {
        // 显示代码单元的原始代码
        htmlContent += `<pre class="code-cell"><code>${escapeHtml(cell.source.join(''))}</code></pre>`;
        // 如果代码单元有输出，解析并显示输出
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

// 返回按钮的跳转方法
const goBack = () => {
  router.push({ path: '/blogs' });
};
</script>

<style>
.blog-detail {
  width: 50%;
  margin: 0 auto;
  padding: 20px;
  text-align: left; /* 确保内容左对齐 */
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
.notebook-content img 
{
  max-width: 100% !important; 
  /* width: 40px; */
  height: auto !important;  /* 保持纵横比 */
  display: block;
  margin: 0 auto; /* 居中 */
}
/* {
  max-width: 100% !important;
  height: auto !important;
  display: block;
  margin: 0 auto;
} */

/* Notebook 中代码块样式 */
.code-cell {
  border: 1px solid #ddd;
  padding: 10px;
  overflow-x: auto;
  border-radius: 5px;
  margin-bottom: 10px;
}

pre {
  white-space: pre-wrap;
}
</style>
