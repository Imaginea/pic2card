<template>
    <div>
        <!-- <h1>Find Similar Images</h1> -->
        <!-- <input type="file" accept="image/*" @change="onChange" /> -->

        <div class="random-images">
            <button v-on:click="reloadRandomImages">Reload Images</button>
            <RandomImageGrid v-if="randomImages" v-bind:data="randomImages" />
        </div>
        <!-- <div class="preview">
            <img v-if="item.imageUrl" :src="item.imageUrl" />
        </div>
        <RandomImageGrid v-if="similarImages" v-bind:data="similarImages" /> -->
    </div>
</template>

<script>
// import UploadImage from 'vue-upload-image'
import axios from 'axios'
import RandomImageGrid from '@/components/RandomImageGrid'
import ImageApi from '@/services/ImageApi.js'

export default {
    name: 'ImageSearch',
    components: {
        RandomImageGrid
    },
    data() {
        return {
            item: {
                image: null,
                imageUrl: null
            },
            // similarImages: null,
            randomImages: null
        }
    },

    beforeMount() {
        this.reloadRandomImages()
    },

    methods: {
        reloadRandomImages() {
            // console.log('Reload Images')
            //const URL = 'http://172.17.0.5:8888/image/random'
            ImageApi.getRandomImageUrls().then(response => {
                this.randomImages = response.data
            })
        }
    }
}
</script>

<style scoped>
.search {
    background-attachment: fixed;
}
.preview {
    display: flex;
}

img {
    border-radius: 10px;
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 10%;
}
</style>
