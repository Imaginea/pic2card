import Vue from 'vue'
import App from './App.vue'
import router from './router'
import store from './store'
import vuetify from './plugins/vuetify'

// import UploadImage from 'vue-upload-image'

Vue.config.productionTip = false

// Vue.component('upload-image', UploadImage)

new Vue({
    router,
    store,
    vuetify,
    render: h => h(App)
}).$mount('#app')
