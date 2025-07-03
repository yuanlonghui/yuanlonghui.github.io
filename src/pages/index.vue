<template>
  <div class="wrapper">
    <div class="info">
      <div class="description">
        <h1 v-html="name"></h1>

        <p v-html="degree"></p>

        <p>
          <a :href="organization_url" target="_blank" v-html="organization"></a>
        </p>

        <div class="section-title">Social</div>
        <div class="social_row">
          <a :href="email_to" target="_blank"><el-icon><Message /></el-icon> Email</a>
          <a :href="scholar" target="_blank"><el-icon><Search /></el-icon> Scholar</a>
          <a :href="github" target="_blank"><el-icon><Star /></el-icon> GitHub</a>
        </div>

        <div class="section-title">Contact</div>
        <div class="contact_row">
          <p>Email: {{ email }}<br>Address: {{ address }}</p>
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
          <p v-for="(intro, idx) in introduction" :key="idx" v-html="intro"></p>
        </div>
      </div>
    </div>
  </div>

  <div class="wrapper">
    <div class="info">
      <div class="description">
        <div class="section-title">News</div>
        <el-timeline style="margin-top: 5px; padding: 0%;">
          <el-timeline-item
            v-for="(activity, index) in news"
            :key="index"
            :timestamp="activity.timestamp"
            type="success"
          >
            <div v-html="activity.content"></div>
          </el-timeline-item>
        </el-timeline>
      </div>
    </div>
  </div>
  <BackToTop />
</template>

<script lang="ts" setup>
import { ref, computed, onMounted } from 'vue'

const src = ref('')
const name = ref('')
const degree = ref('')
const organization = ref('')
const organization_url = ref('')
const email = ref('')
const address = ref('')
const scholar = ref('')
const github = ref('')
const introduction = ref<string[]>([])
const news = ref<{ timestamp: string; content: string }[]>([])

const email_to = computed(() => 'mailto:' + email.value)

onMounted(async () => {
  const [aboutRes, newsRes] = await Promise.all([
    fetch('/custom/about.json'),
    fetch('/custom/news.json')
  ])

  const aboutData = await aboutRes.json()
  const newsData = await newsRes.json()

  src.value = aboutData.photo
  name.value = aboutData.name
  degree.value = aboutData.degree
  organization.value = aboutData.organization
  organization_url.value = aboutData.organization_url
  email.value = aboutData.email
  address.value = aboutData.address
  scholar.value = aboutData.scholar
  github.value = aboutData.github
  introduction.value = aboutData.introduction
  news.value = newsData.news
})
</script>

<style scoped>
.wrapper {
  display: flex;
  justify-content: center;
  margin-bottom: 0px;
}

.info {
  display: flex;
  align-items: center;
  width: 50%;
  max-width: 800px;
  flex-wrap: wrap;
}

.description {
  flex: 1;
  text-align: left;
  font-family: "Arial", sans-serif;
}

.photo {
  width: 40%;
  height: auto;
}

.photo img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

h1 {
  font-weight: bold;
  margin-bottom: 8px;
}

p {
  margin: 5px 0;
}

a {
  color: #2c92ff;
  text-decoration: none;
  font-weight: 500;
}

a:hover {
  text-decoration: underline;
}

.section-title {
  font-weight: bold;
  margin-top: 10px;
  margin-bottom:0px;
}

.social_row,
.contact_row {
  margin-top: 5px;
  margin-bottom: 10px;
}

.social_row a {
  display: inline-flex;
  align-items: center;
  margin-right: 15px;
}

.social_row a el-icon {
  margin-right: 5px;
}

@media (max-width: 768px) {
  .info {
    width: 95%;
  }
}
</style>
