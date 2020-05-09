<template>
    <div>
        <h1>Similar Images</h1>
        <template>
            <div class="gallery">
                <div
                    class="gallery-panel"
                    v-for="photo in data"
                    :key="photo.doc_id + 1"
                >
                    <img
                        :src="imageUrl(photo.image_url)"
                        :key="photo.doc_id + 1"
                    />

                    <p>Distance: {{ photo.dist }}</p>
                </div>
            </div>
        </template>
    </div>
</template>

<script>
import ImageApi from '@/services/ImageApi'

export default {
    name: 'ImageGrid',
    props: ['data'],
    methods: {
        imageUrl(path) {
            //console.log('nested path: ' + path)
            return 'http://172.17.0.5:8888' + path
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
    background-color: bisque;
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
