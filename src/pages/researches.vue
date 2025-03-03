<template>
  <div class="wrapper">
    <div class="info">
      <div class="description">
        <div class="section-title"><h1>Publication</h1></div>
        <hr />
        <div class="introduction">
          <div v-for="pub in publications">
            <Paper 
            :title="pub.title"
            :conference="pub.conference"
            :authors="pub.authors"
            :introduction="pub.introduction"
            :paper_link="pub.paper_link"
            :code_link="pub.code_link"
            :img_link="pub.img_link"></Paper>
            <hr />
          </div>
        </div>
      </div>
    </div>
  </div>
  <div class="wrapper">
    <div class="info">
      <div class="description">
        <div class="section-title"><h1>Preprints</h1></div>
        <hr />
        <div class="introduction">
          <div v-for="pub in preprints">
            <Paper 
            :title="pub.title"
            :conference="pub.conference"
            :authors="pub.authors"
            :introduction="pub.introduction"
            :paper_link="pub.paper_link"
            :code_link="pub.code_link"
            :img_link="pub.img_link"></Paper>
            <hr />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { computed } from '@vue/reactivity';
import { ref, onMounted } from 'vue';

const publications = ref()
const preprints = ref()

onMounted(async () => {
  const response = await fetch('/custom/researches.json');
  const userData = await response.json();
  publications.value = userData.publications
  preprints.value = userData.preprints
});
</script>

<style lang="css">
/* 外部容器 */
.wrapper {
  display: flex;
  justify-content: center;
}

/* 信息容器 */
.info {
  display: flex;
  align-items: center;
  width: 50%;
  max-width: 800px;
}

/* 个人介绍部分 */
.description {
  flex: 1;
  text-align: left;
  font-family: "Arial", sans-serif;
}


/* 标题样式 */
h1 {
  /* font-size: 24px; */
  font-weight: bold;
  margin-bottom: 8px;
}

/* 普通文本 */
p {
  /* font-size: 14px; */
  margin: 5px 0;
}

/* 链接样式 */
a {
  color: #2c92ff;
  text-decoration: none;
  font-weight: 500;
}

a:hover {
  color: #2c92ff;
  text-decoration: underline;
}

/* 分区标题 */
.section-title {
  font-weight: bold;
  /* font-size: 16px; */
  margin-top: 10px;
}

/* 间距微调 */
.social_row, .contact_row {
  margin-bottom: 10px;
}

/* 调整图标和文字的对齐 */
.social_row p a, .contact_row p {
  display: inline-flex;
  align-items: center;
  margin-right: 15px;
}

.social_row p a el-icon {
  margin-right: 5px;
}
</style>
