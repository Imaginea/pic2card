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
                    contain=""
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
import AdaptiveCardApi from '@/services/ImageApi.js'
import axios from 'axios'

export default {
    name: 'Pic2Card',
    props: {
        base64_image: String
    },
    data() {
        return {
            image_str: this.base64_image,
            card_html: null,
            card_loading: true
        }
    },
    computed: {
        image_data_url() {
            return 'data:image/png;base64,' + this.image_str
        }
    },
    methods: {
        pic2Card(base64_image) {
            axios({
                method: 'post',
                url: 'http://172.17.0.5:5050/predict_json',
                timeout: 200000,
                data: { image: base64_image },
                headers: {
                    'Content-Type': 'application/json'
                }
            }).then(response => {
                console.log(response.data)
                let card_json = response.data['card_json']
                // Initialize the adaptive card.
                let adaptiveCard = new AdaptiveCards.AdaptiveCard()
                adaptiveCard.parse(card_json)
                this.card_html = adaptiveCard.render()
                this.$refs.cards.appendChild(this.card_html)
            })
        },
        renderCard() {
            console.log('asdfasdf')
            AdaptiveCardApi.getAdaptiveCard(this.image_str).then(response => {
                console.log(response.data)
                this.card_json = response.data['card_json']
            })
        }
    },
    mounted() {
        /**
         * Called after dom got attached.
         */
        this.$refs.cards.innerHTML = ''
        if (this.card_html) {
            this.$refs.cards.appendChild(this.card_html)
        }
    },
    beforeMount() {
        // Before dom attachment.
        // this.renderCard()
        this.pic2Card(this.image_str)
    }
}
</script>

<style scoped></style>
