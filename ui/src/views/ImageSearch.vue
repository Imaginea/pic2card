<template>
    <div class="image-search">
        <!-- <h1>Image Similarity ...</h1> -->
        <!-- <ImageSearch /> -->
        <VImageGrid v-bind:gname="randomName" v-bind:images="randomImages" />
        <!-- <VImageGrid /> -->
    </div>
</template>

<script>
import ImageSearch from '@/components/ImageSearch.vue'
import VImageGrid from '@/components/VImageGrid.vue'
import ImageApi from '@/services/ImageApi.js'

export default {
    name: 'ImageSearchView',
    components: {
        VImageGrid
        // ImageSearch
    },
    data() {
        return {
            randomImages: null,
            randomName: 'Random Images'
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

<style scoped></style>
