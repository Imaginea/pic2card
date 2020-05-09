<template>
    <v-container fluid>
        <v-row>
            <v-col cols="4" sm="6" class="d-flex child-flex">
                <v-card color="blue lighten-5 dark">
                    <v-container fluid>
                        <v-row>
                            <v-col>
                                <p class="text-center subtitle-5">
                                    Query Images, select an image to find images
                                    similar to query image.
                                </p>
                            </v-col>
                        </v-row>
                        <v-row>
                            <v-col>
                                <v-btn
                                    color="black"
                                    block
                                    text
                                    rounded
                                    :loading="isLoading"
                                    @click="reloadRandomImages"
                                >
                                    Reload
                                </v-btn>
                            </v-col>
                        </v-row>
                        <v-row>
                            <v-col
                                v-for="image in randomImages"
                                :key="image.doc_id"
                                class="d-flex child-flex"
                                cols="6"
                                @click="findSimilarImages(image.doc_id)"
                            >
                                <v-card tile class="d-flex" link ripple>
                                    <v-img
                                        :src="`${imageUrl(image.image_url)}`"
                                        :lazy-src="
                                            `${imageUrl(image.image_url)}`
                                        "
                                        aspect-ratio="1"
                                        class="grey lighten-2"
                                        contain
                                    >
                                        <template v-slot:placeholder>
                                            <v-row
                                                class="fill-height ma-0"
                                                align="center"
                                                justify="center"
                                            >
                                                <v-progress-circular
                                                    indeterminate
                                                    color="grey lighten-5"
                                                ></v-progress-circular>
                                            </v-row>
                                        </template>
                                    </v-img>
                                </v-card>
                            </v-col>
                        </v-row>
                    </v-container>
                </v-card>
            </v-col>

            <v-col cols="6" cols-sm="6" class="d-flex child-flex">
                <v-card color="blue lighten-5 dark" loading>
                    <v-container fluid>
                        <v-row>
                            <v-col>
                                <p class="text-center subtitle-2">
                                    Similar Images
                                </p>
                            </v-col>
                        </v-row>
                        <v-row>
                            <v-col
                                v-for="image in similarImages"
                                :key="image.doc_id"
                                class="d-flex child-flex"
                                cols="6"
                            >
                                <v-card class="d-flex">
                                    <v-img
                                        :src="`${imageUrl(image.image_url)}`"
                                        :lazy-src="
                                            `${imageUrl(image.image_url)}`
                                        "
                                        class="align-end"
                                        contain
                                    >
                                        <v-card-subtitle
                                            v-text="image.dist.toFixed(2)"
                                        ></v-card-subtitle>
                                        <template v-slot:placeholder>
                                            <v-row
                                                class="fill-height ma-0"
                                                align="center"
                                                justify="center"
                                            >
                                                <v-progress-circular
                                                    indeterminate
                                                    color="grey lighten-5"
                                                ></v-progress-circular>
                                            </v-row>
                                        </template>
                                    </v-img>
                                </v-card>
                            </v-col>
                        </v-row>
                    </v-container>
                </v-card>
            </v-col>
        </v-row>
    </v-container>
</template>

<script>
import ImageApi from '@/services/ImageApi.js'

export default {
    name: 'VGridImage',
    props: ['images'],
    data() {
        return {
            isLoading: false,
            randomImages: null,
            similarImages: null
        }
    },
    beforeMount() {
        this.reloadRandomImages()
    },
    methods: {
        imageUrl(path) {
            // Reset the similar images, as reloaded the input images
            //console.log('path: ' + path)
            // this.similarImages = null

            return 'http://172.17.0.5:8888' + path
        },
        reloadRandomImages() {
            // console.log('Reload Images')
            //const URL = 'http://172.17.0.5:8888/image/random'
            this.isLoading = true
            ImageApi.getRandomImageUrls(4).then(response => {
                this.randomImages = response.data

                // Reset the result page.
                this.similarImages = null
                this.isLoading = false
            })
        },
        findSimilarImages(imageId) {
            // console.log('reload: ' + imageId)
            ImageApi.getSimilarImages(imageId).then(response => {
                // console.log(response)
                this.similarImages = response.data
            })
        }
    }
}
</script>

<style></style>
