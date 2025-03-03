<template>
  <el-config-provider namespace="ep">
    <BaseHeader />
    <div class="main-container flex">
      <div w="full" py="4">
        <RouterView />
      </div>
    </div>
  </el-config-provider>
</template>

<style>
#app {
  text-align: center;
  color: var(--ep-text-color-primary);
}

.main-container {
  height: calc(100vh - var(--ep-menu-item-height) - 4px);
}

/* .main-container {
  height: calc(100vh - var(--ep-menu-item-height) - 4px);
  overflow-y: auto; 
  scrollbar-width: none; 
}

.main-container::-webkit-scrollbar {
  display: none; 
} */
</style>
