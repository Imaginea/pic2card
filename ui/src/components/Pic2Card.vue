<template>
    <v-container>
        <v-row>
            <v-col>
                <p ref="cards"></p>
            </v-col>
            <v-col>
                <v-img
                    :src="image_data_url"
                    lazy-src="https://picsum.photos/510/300?random"
                    aspect-ratio="1"
                    class="grey lighten-2"
                    max-width="500"
                    max-height="300"
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
            </v-col>
        </v-row>
    </v-container>
</template>

<script>
import * as AdaptiveCards from 'adaptivecards'
import getAdaptiveCard from '@/services/ImageApi.js'

export default {
    name: 'Pic2Card',
    props: ['base64_image'],
    data() {
        return {
            image_str: this.base64_image,
            card_json: null
        }
    },
    computed: {
        image_data_url() {
            //console.log('base64 iamge: ', this.base64_image)
            return 'data:image/png;base64,' + this.image_str
            // return this.base64_image
        }
    },
    methods: {
        pic2Card(base64_image) {
            getAdaptiveCard(base64_image).then(response => {
                console.log(response.data)
                this.card_json = response.data
            })
        },
        renderCard() {
            console.log('asdfasdf')
            this.pic2Card(this.image_str)

            // let adaptiveCard = new AdaptiveCards.AdaptiveCard()
            // adaptiveCard.parse(this.renderedCard)
            // let cardHTML = adaptiveCard.render()
            // this.renderedCard = cardHTML
        }
    }
    // mounted() {
    //     /**
    //      * Called after dom got attached.
    //      */
    //     // this.$refs.cards.innerHTML = ''
    //     // this.$refs.cards.appendChild(this.renderedCard)
    // },
    // beforeMount() {
    //     // Before dom attachment.
    //     this.renderCard()
    // }
}
</script>

<style scoped></style>
