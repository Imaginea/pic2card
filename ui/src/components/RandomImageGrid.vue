<template>
    <div>
        <h1>Random Images</h1>

        <p>Click on an image to see matching similar images</p>
        <template>
            <div class="gallery">
                <div
                    class="gallery-panel"
                    v-for="photo in data"
                    :key="photo.doc_id"
                >
                    <a @click="onImageSelect(photo.doc_id)" href="#">
                        <img
                            :id="photo.doc_id"
                            :src="imageUrl(photo.image_url)"
                        />
                    </a>
                    <!-- <p>Distance: {{ photo.dist }}</p> -->
                </div>
            </div>

            <ImageGrid v-if="similarImages" v-bind:data="similarImages" />
        </template>
    </div>
</template>

<script>
import ImageGrid from '@/components/ImageGrid'
import ImageApi from '@/services/ImageApi'

export default {
    name: 'RandomGrid',
    components: {
        ImageGrid
    },
    props: ['data'],

    data() {
        return {
            similarImages: null
        }
    },

    methods: {
        // imageUrl(path) {
        //     return ImageApi.baseURL() + path
        // },
        onImageSelect(doc_id) {
            //console.log('image selected: ' + doc_id)
            ImageApi.getSimilarImages(doc_id).then(response => {
                //console.log(response)
                this.similarImages = response.data
            })
        },
        imageUrl(path) {
            // Reset the similar images, as reloaded the input images
            //console.log('path: ' + path)
            // this.similarImages = null

            return 'http://172.17.0.5:8888' + path
        },
        getSimilarImg(docId) {
            //console.log('get similar images')
            return 'http://172.17.0.5:8888/image/similar'
        }
    }
}
</script>

<style scoped>
.gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(20rem, 1fr));
    grid-gap: 1rem;
    max-width: 80rem;
    margin: 1rem auto;
    padding: 0 5rem;
    background-color: azure;
}
.gallery-panel img {
    width: 50%;
    /* height: 22vw; */
    object-fit: cover;
    border-radius: 0.75rem;
}

img {
    padding: 1rem;
}
</style>
