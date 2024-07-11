const { addKeyword, EVENTS } = require("@bot-whatsapp/bot");
const { createEvent } = require("../scripts/calendar")

const formFlow = addKeyword(EVENTS.ACTION)
    .addAnswer("Excelente! Voy a agendarte el evento, ¿Podrías especificar qué tipo de evento es?", { capture: true },
        async (ctx, ctxFn) => {
            await ctxFn.state.update({ name: ctx.body }); // Guarda el tipo de evento que ingreses
        }
    )
    .addAnswer("Perfecto, Si hay algún otro detalle relevante para este evento, por favor indícalo.", { capture: true },
        async (ctx, ctxFn) => {
            await ctxFn.state.update({ motive: ctx.body }); // Guarda el motivo en el estado
        }
    )
    .addAnswer("¡Listo! He registrado el evento correctamente.", null,
        async (ctx, ctxFn) => {
            const userInfo = await ctxFn.state.getMyState();
            const eventName = userInfo.name;
            const description = userInfo.motive;
            const date = userInfo.date;
            const eventId = await createEvent(eventName, description, date)
            await ctxFn.state.clear();
        }
    )

module.exports = { formFlow };