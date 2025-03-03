<template>
    <div class="paper-card">
        <!-- 论文标题 -->
        <h2>
        <strong>{{ title }}</strong>
        </h2>

        <!-- 会议信息 -->
        <p class="conference">{{ conference }}</p>

        <!-- 作者信息 -->
        <p class="authors">{{ authors }}</p>

        <!-- 论文配图 -->
        <div class="paper-image">
        <img v-if="img_link" :src="img_link" alt="Paper Image" />
        </div>

        <!-- 论文介绍 -->
        <p class="introduction">{{ introduction }}</p>

        <!-- 论文和代码链接 -->
        <div class="links">
        <a v-if="paper_link" :href="paper_link" target="_blank">Paper</a>
        &nbsp;
        <a v-if="code_link" :href="code_link" target="_blank">Code</a>
        </div>
    </div>
</template>

<script lang="ts" setup>
defineProps<{
title: string;
conference: string;
authors: string;
introduction: string;
paper_link?: string;
code_link?: string;
img_link?: string;
}>();
</script>

<style scoped>
.paper-card {
width: 100%;
}

h2 {
font-weight: bold;
margin-bottom: 10px;
}

.conference {
color: red;
font-weight: bold;
}

.authors {
margin-bottom: 10px;
}

.paper-image img {
width: 100%;
height: auto;
border-radius: 5px;
}

.introduction {
line-height: 1.5;
}

.links {
margin-top: 5px;
}

.links a {
color: #39c;
text-decoration: none;
font-weight: bold;
}

.links a:hover {
color: #069;
}
</style>
