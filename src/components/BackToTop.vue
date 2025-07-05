<template>
  <transition name="fade">
    <el-icon
      v-show="visible"
      class="back-to-top"
      @click="scrollToTop"
      title="回到顶部"
    >
      <!-- <Location /> -->
       <img src="../assets/up-arrow.png" style="width: 50px; height: auto;"/>
    </el-icon>
  </transition>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { Top } from '@element-plus/icons-vue'

const visible = ref(false)

const handleScroll = () => {
  visible.value = window.scrollY > 300
}

const scrollToTop = () => {
  window.scrollTo({ top: 0, behavior: 'smooth' })
}

onMounted(() => {
  window.addEventListener('scroll', handleScroll)
})

onUnmounted(() => {
  window.removeEventListener('scroll', handleScroll)
})
</script>

<style scoped>
.back-to-top {
  position: fixed;
  top: 30px;
  right: 30px;
  font-size: 32px;
  color: #2c92ff;
  cursor: pointer;
  z-index: 9999;
  transition: transform 0.3s ease, opacity 0.3s ease;
}

.back-to-top:hover {
  transform: scale(1.2) rotate(15deg);
  color: #1a73e8;
}

/* 动画淡入淡出 */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
