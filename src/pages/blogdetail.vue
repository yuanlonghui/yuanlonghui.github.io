<template>
    <div class="blog-detail">
      <el-button @click="goBack" class="back-button">返回</el-button>
      <div v-html="renderedMarkdown" class="markdown-content markdown-body"></div>
    </div>
  </template>
  
<script lang="ts" setup>
import { ref, onMounted } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import MarkdownIt from 'markdown-it';
import markdownItMathjax3 from 'markdown-it-mathjax3';

const route = useRoute();
const router = useRouter();
const renderedMarkdown = ref('');
const mdParser = new MarkdownIt({ html: true }).use(markdownItMathjax3);

onMounted(async () => {
  const mdPath = route.query.path as string;
  if (mdPath) {
    try {
      const response = await fetch(mdPath);
      const text = await response.text();
      renderedMarkdown.value = mdParser.render(text);
      console.log(renderedMarkdown.value)
    } catch (error) {
      renderedMarkdown.value = '<p>无法加载博客内容</p>';
    }
  }
});

const goBack = () => {
  router.push({ name: '/blogs' });
};
</script>

<style>
.blog-detail {
  /* max-width: 800px; */
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
  /* background: #fff; */
}

.markdown-body img {
  max-width: 100% !important; 
  /* width: 40px; */
  height: auto !important;  /* 保持纵横比 */
  display: block;
  margin: 0 auto; /* 居中 */
}
</style>
  