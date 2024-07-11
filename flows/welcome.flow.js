const { addKeyword, EVENTS } = require('@bot-whatsapp/bot');

const welcomeFlow = addKeyword(EVENTS.ACTION)
    .addAction(async (ctx, ctxFn) => {
        await ctxFn.endFlow("¡Hola! ¿En qué puedo asistirte hoy?")
    })

module.exports = {welcomeFlow};