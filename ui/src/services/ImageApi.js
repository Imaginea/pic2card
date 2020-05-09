import axios from 'axios'

const apiClient = axios.create({
    baseURL: 'http://172.17.0.5:5050',
    withCredentials: false,
    headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json'
    },
    timeout: 10000
})

export default {
    baseURL() {
        return apiClient.baseURL
    },
    getTemplateImages() {
        return apiClient.get('/get_card_templates')
    },
    getAdaptiveCard(base64_image) {
        let data = {
            image: base64_image
        }
        config = {
            header: {
                'Content-Type': 'image/png'
            }
        }
        return apiClient.post('/predict_json', data)
    }
}
