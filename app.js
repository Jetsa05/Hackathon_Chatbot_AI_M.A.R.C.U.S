const { createBot, createProvider, createFlow, addKeyword, EVENTS } = require('@bot-whatsapp/bot');
const QRPortalWeb = require('@bot-whatsapp/portal');
const BaileysProvider = require('@bot-whatsapp/provider/baileys');
const MockAdapter = require('@bot-whatsapp/database/mock');
const { chat } = require("./scripts/chatgpt.js"); // Importa tu función para interactuar con ChatGPT
const { welcomeFlow } = require('./flows/welcome.flow.js');
const { formFlow } = require("./flows/form.flow.js");
const { dateFlow, confirmationFlow } = require("./flows/date.flow.js");

const flowPrincipal = addKeyword(EVENTS.WELCOME)
    .addAction(async (ctx, ctxFn) => {
        const bodyText = ctx.body.toLowerCase();

        // El usuario está saludando?
        const keywordsSaludo = ["hola", "buenas", "ola"];
        const containsKeywordSaludo = keywordsSaludo.some(keyword => bodyText.includes(keyword));
        if (containsKeywordSaludo && ctx.body.length < 8) {
            return await ctxFn.gotoFlow(welcomeFlow); // Si está saludando, ir al flujo de bienvenida
        }

        // El usuario quiere agendar una cita o realizar alguna acción específica
        const keywordsAgendar = ["agendar", "agregar", "configurar"];
        const containsKeywordAgendar = keywordsAgendar.some(keyword => bodyText.includes(keyword));
        if (containsKeywordAgendar) {
            return ctxFn.gotoFlow(dateFlow); // Si quiere agendar una cita, ir al flujo de fechas
        }

        // Si no se comprende la solicitud, usar ChatGPT para intentar responder
        const prompt = "Eres un chat bot que es un asistente universitario personal te llamas StudyAid";
        const conversations = [];
        const contextMessages = conversations.flatMap(conv => [
            { role: "user", content: conv.question },
            { role: "assistant", content: conv.answer }
        ]);
        contextMessages.push({ role: "user", content: ctx.body });

        // Obtener respuesta de ChatGPT
        const response = await chat(prompt, contextMessages);

        // Enviar la respuesta al usuario
        await ctxFn.flowDynamic(response);
    });

const main = async () => {
    const adapterDB = new MockAdapter();
    const adapterFlow = createFlow([flowPrincipal, welcomeFlow, formFlow, dateFlow, confirmationFlow]);
    const adapterProvider = createProvider(BaileysProvider);

    createBot({
        flow: adapterFlow,
        provider: adapterProvider,
        database: adapterDB,
    });

    QRPortalWeb();
};

main();
