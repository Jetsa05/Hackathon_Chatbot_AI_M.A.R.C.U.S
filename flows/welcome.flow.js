const { addKeyword, EVENTS } = require('@bot-whatsapp/bot');

const welcomeFlow = addKeyword(EVENTS.ACTION)
    .addAction(async (ctx, ctxFn) => {
        await ctxFn.endFlow("Bienvenido a este chatbot! \nPodes escribir 'Agendar cita' para reservar una cita")
    })

module.exports = { welcomeFlow };