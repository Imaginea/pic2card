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
                    contain
                >
                    <template v-slot:placeholder>
                        <v-row class="fill-height ma-0" align="center" justify="center">
                            <v-progress-circular indeterminate color="grey lighten-5"></v-progress-circular>
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
                adaptiveCard.hostConfig = new AdaptiveCards.HostConfig({
                    spacing: {
                        small: 3,
                        default: 8,
                        medium: 20,
                        large: 30,
                        extraLarge: 40,
                        padding: 10
                    },
                    separator: {
                        lineThickness: 1,
                        lineColor: '#EEEEEE'
                    },
                    supportsInteractivity: true,
                    fontTypes: {
                        default: {
                            fontFamily:
                                "'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif",
                            fontSizes: {
                                small: 12,
                                default: 14,
                                medium: 17,
                                large: 21,
                                extraLarge: 26
                            },
                            fontWeights: {
                                lighter: 200,
                                default: 400,
                                bolder: 600
                            }
                        },
                        monospace: {
                            fontFamily: "'Courier New', Courier, monospace",
                            fontSizes: {
                                small: 12,
                                default: 14,
                                medium: 17,
                                large: 21,
                                extraLarge: 26
                            },
                            fontWeights: {
                                lighter: 200,
                                default: 400,
                                bolder: 600
                            }
                        }
                    },
                    containerStyles: {
                        default: {
                            backgroundColor: '#FFFFFF',
                            foregroundColors: {
                                default: {
                                    default: '#000000',
                                    subtle: '#767676'
                                },
                                accent: {
                                    default: '#0063B1',
                                    subtle: '#0063B1'
                                },
                                attention: {
                                    default: '#FF0000',
                                    subtle: '#DDFF0000'
                                },
                                good: {
                                    default: '#54a254',
                                    subtle: '#DD54a254'
                                },
                                warning: {
                                    default: '#c3ab23',
                                    subtle: '#DDc3ab23'
                                }
                            }
                        },
                        emphasis: {
                            backgroundColor: '#F0F0F0',
                            foregroundColors: {
                                default: {
                                    default: '#000000',
                                    subtle: '#767676'
                                },
                                accent: {
                                    default: '#2E89FC',
                                    subtle: '#882E89FC'
                                },
                                attention: {
                                    default: '#FF0000',
                                    subtle: '#DDFF0000'
                                },
                                good: {
                                    default: '#54a254',
                                    subtle: '#DD54a254'
                                },
                                warning: {
                                    default: '#c3ab23',
                                    subtle: '#DDc3ab23'
                                }
                            }
                        },
                        accent: {
                            backgroundColor: '#C7DEF9',
                            foregroundColors: {
                                default: {
                                    default: '#333333',
                                    subtle: '#EE333333'
                                },
                                dark: {
                                    default: '#000000',
                                    subtle: '#66000000'
                                },
                                light: {
                                    default: '#FFFFFF',
                                    subtle: '#33000000'
                                },
                                accent: {
                                    default: '#2E89FC',
                                    subtle: '#882E89FC'
                                },
                                attention: {
                                    default: '#cc3300',
                                    subtle: '#DDcc3300'
                                },
                                good: {
                                    default: '#54a254',
                                    subtle: '#DD54a254'
                                },
                                warning: {
                                    default: '#e69500',
                                    subtle: '#DDe69500'
                                }
                            }
                        },
                        good: {
                            backgroundColor: '#CCFFCC',
                            foregroundColors: {
                                default: {
                                    default: '#333333',
                                    subtle: '#EE333333'
                                },
                                dark: {
                                    default: '#000000',
                                    subtle: '#66000000'
                                },
                                light: {
                                    default: '#FFFFFF',
                                    subtle: '#33000000'
                                },
                                accent: {
                                    default: '#2E89FC',
                                    subtle: '#882E89FC'
                                },
                                attention: {
                                    default: '#cc3300',
                                    subtle: '#DDcc3300'
                                },
                                good: {
                                    default: '#54a254',
                                    subtle: '#DD54a254'
                                },
                                warning: {
                                    default: '#e69500',
                                    subtle: '#DDe69500'
                                }
                            }
                        },
                        attention: {
                            backgroundColor: '#FFC5B2',
                            foregroundColors: {
                                default: {
                                    default: '#333333',
                                    subtle: '#EE333333'
                                },
                                dark: {
                                    default: '#000000',
                                    subtle: '#66000000'
                                },
                                light: {
                                    default: '#FFFFFF',
                                    subtle: '#33000000'
                                },
                                accent: {
                                    default: '#2E89FC',
                                    subtle: '#882E89FC'
                                },
                                attention: {
                                    default: '#cc3300',
                                    subtle: '#DDcc3300'
                                },
                                good: {
                                    default: '#54a254',
                                    subtle: '#DD54a254'
                                },
                                warning: {
                                    default: '#e69500',
                                    subtle: '#DDe69500'
                                }
                            }
                        },
                        warning: {
                            backgroundColor: '#FFE2B2',
                            foregroundColors: {
                                default: {
                                    default: '#333333',
                                    subtle: '#EE333333'
                                },
                                dark: {
                                    default: '#000000',
                                    subtle: '#66000000'
                                },
                                light: {
                                    default: '#FFFFFF',
                                    subtle: '#33000000'
                                },
                                accent: {
                                    default: '#2E89FC',
                                    subtle: '#882E89FC'
                                },
                                attention: {
                                    default: '#cc3300',
                                    subtle: '#DDcc3300'
                                },
                                good: {
                                    default: '#54a254',
                                    subtle: '#DD54a254'
                                },
                                warning: {
                                    default: '#e69500',
                                    subtle: '#DDe69500'
                                }
                            }
                        }
                    },
                    imageSizes: {
                        small: 40,
                        medium: 80,
                        large: 160
                    },
                    actions: {
                        maxActions: 5,
                        spacing: 'default',
                        buttonSpacing: 8,
                        showCard: {
                            actionMode: 'inline',
                            inlineTopMargin: 8
                        },
                        actionsOrientation: 'horizontal',
                        actionAlignment: 'stretch'
                    },
                    adaptiveCard: {
                        allowCustomStyle: false
                    },
                    imageSet: {
                        imageSize: 'medium',
                        maxImageHeight: 100
                    },
                    factSet: {
                        title: {
                            color: 'default',
                            size: 'default',
                            isSubtle: false,
                            weight: 'bolder',
                            wrap: true,
                            maxWidth: 150
                        },
                        value: {
                            color: 'default',
                            size: 'default',
                            isSubtle: false,
                            weight: 'default',
                            wrap: true
                        },
                        spacing: 8
                    }
                })
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
