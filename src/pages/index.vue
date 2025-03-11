<template>
  <div class="wrapper">
    <div class="info">
      <div class="description">
        <div class="name">
          <h1>{{ name }}</h1>
        </div>
        <div class="degree">
          <p>{{ degree }}</p>
        </div>
        <div class="organization">
          <p><a :href="organization_url" target="_blank">{{ organization }}</a></p>
        </div>
        <div class="section-title">Social</div>
        <div class="social_row">
          <p>
            <a :href="email_to" target="_blank"><el-icon><Message /></el-icon> Email</a>
            <a :href="scholar" target="_blank"><el-icon><Search /></el-icon> Scholar</a>
            <a :href="github" target="_blank"><el-icon><Star /></el-icon>GitHub</a>
          </p>
        </div>
        <div class="section-title">Contact</div>
        <div class="contact_row">
          <p>
            Email: {{ email }}<br>
            Address: {{ address }}
          </p>
        </div>
      </div>
      <div class="photo">
        <el-image :src="src" />
      </div>
    </div>
  </div>
  <div class="wrapper">
    <div class="info">
      <div class="description">
        <div class="section-title">About Me</div>
        <div class="introduction">
          <p v-for="intro in introduction">{{ intro }}</p>
        </div>
      </div>
    </div>
  </div>
  <div class="wrapper">
    <div class="info">
      <div class="description">
        <div class="section-title">News</div>
        <div style="margin-top: 5px;">
          <el-timeline style="padding: 0%;">
            <el-timeline-item
              v-for="(activity, index) in news"
              :key="index"
              :timestamp="activity.timestamp"
            >
              {{ activity.content }}
            </el-timeline-item>
          </el-timeline>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { computed } from '@vue/reactivity';
import { ref, onMounted } from 'vue';

const src = ref('');
const name = ref('');
const degree = ref('');
const organization = ref('');
const organization_url = ref('');
const email = ref('');
const address = ref('');
const email_to = computed(() => "mailto:" + email.value);
const scholar = ref('');
const github = ref('');
const introduction = ref()
const news = ref()

onMounted(async () => {
  let response = await fetch('/custom/about.json');
  let userData = await response.json();
  src.value = userData.photo;
  name.value = userData.name;
  degree.value = userData.degree;
  organization.value = userData.organization;
  organization_url.value = userData.organization_url;
  email.value = userData.email;
  address.value = userData.address;
  scholar.value = userData.scholar;
  github.value = userData.github;
  introduction.value = userData.introduction
});

onMounted(async ()=> {
  let response = await fetch('/custom/news.json');
  let userData = await response.json();
  news.value = userData.news
  console.log(news.value)
})

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

/* 头像 */
.photo {
  width: 40%; /* 图片宽度可根据需要调整 */
  height: auto; /* 保持图片的纵横比 */
}

.photo img {
  width: 100%;
  height: 100%;
  object-fit: cover;
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

@media (max-width: 768px) {
  .info {
    width: 95%;  /* 手机设备下占据95%的宽度 */
  }
}
</style>
