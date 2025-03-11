<template>
    <div class="wrapper">
      <div class="info">
        <div class="description">
          <div class="section-title"><h1>Blogs</h1></div>
          <div style="margin-top: 5px;">
            <el-timeline style="padding: 0%;">
              <el-timeline-item
                v-for="(activity, index) in blogs"
                :key="index"
                :timestamp="activity.timestamp"
                placement="top">
                <el-card shadow="hover" class="cardstyle" @click="goToBlog(activity.path)">
                  <h2>{{ activity.content }}</h2>
                </el-card>
              </el-timeline-item>
            </el-timeline>
          </div>
        </div>
      </div>
    </div>
</template>
  
<script lang="ts" setup>
import { computed } from '@vue/reactivity';
import { ElCard } from 'element-plus';
import { ref, onMounted } from 'vue';
import { useRoute, useRouter } from 'vue-router';

const route = useRoute();
const router = useRouter();

const blogs = ref()

onMounted(async () => {
  let response = await fetch('/custom/blogs.json');
  let userData = await response.json();
  blogs.value = userData.blogs
});

const goToBlog = (mdPath: string) => {
  console.log("路径：", mdPath)
  router.push({ name: '/blogdetail', query: { path: mdPath } });
};

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
margin-bottom: 10px;
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

.cardstyle {
  border: 1px solid #9e9b9b54;
  transition: box-shadow 0.3s ease, transform 0.2s ease;
  border-radius: 8px;
  overflow: hidden;
  padding: 10px;
}

.cardstyle:hover {
  box-shadow: 0 4px 10px rgba(255, 0, 0, 0.425);
  transform: translateY(-2px);
}

@media (max-width: 768px) {
  .info {
    width: 95%;  /* 手机设备下占据95%的宽度 */
  }
}
</style>
