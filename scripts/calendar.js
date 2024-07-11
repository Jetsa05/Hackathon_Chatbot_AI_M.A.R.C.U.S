const { google } = require('googleapis');

// Inicializa la librería cliente de Google y configura la autenticación con credenciales de la cuenta de servicio.
const auth = new google.auth.GoogleAuth({
    keyFile: './google.json',  // Ruta al archivo de clave de tu cuenta de servicio.
    scopes: ['https://www.googleapis.com/auth/calendar']  // Alcance para la API de Google Calendar.
});

const calendar = google.calendar({ version: "v3" });

// Constantes configurables
const calendarID = 'Tu calendar ID';
const timeZone = 'Europe/London';
//'Europe/London';
// Ejemplos de zonas horarias:
// UTC: 'Etc/UTC'
// Argentina: America/Argentina/Buenos_Aires'
// Colombia: 'America/Bogota'
// México: 'America/Mexico_City'
// Perú: 'America/Lima'
// Uruguay: 'America/Montevideo'
// El Salvador: 'America/El_Salvador'

const rangeLimit = {
    days: [1, 2, 3, 4, 5], // Lunes a Viernes
    startHour: 9,
    endHour: 18
};
const standardDuration = 1; // Duración por defecto de las citas (1 hora)
const dateLimit = 30; // Maximo de dias a traer la lista de Next Events

/**
 * Crea un evento en el calendario.
 * @param {string} eventName - Nombre del evento.
 * @param {string} description - Descripción del evento.
 * @param {string} date - Fecha y hora de inicio del evento en formato ISO (e.g., '2024-06-01T10:00:00-07:00').
 * @param {number} [duration=standardDuration] - Duración del evento en horas. Default es 1 hora.
 * @returns {string} - URL de la invitación al evento.
 */
async function createEvent(eventName, description, date, duration = standardDuration) {
    try {
        // Autenticación
        const authClient = await auth.getClient();
        google.options({ auth: authClient });

        // Fecha y hora de inicio del evento
        const startDateTime = new Date(date);
        // Fecha y hora de fin del evento
        const endDateTime = new Date(startDateTime);
        endDateTime.setHours(startDateTime.getHours() + duration);

        const event = {
            summary: eventName,
            description: description,
            start: {
                dateTime: startDateTime.toISOString(),
                timeZone: timeZone,
            },
            end: {
                dateTime: endDateTime.toISOString(),
                timeZone: timeZone,
            },
            colorId: '2' // El ID del color verde en Google Calendar es '11'
        };

        const response = await calendar.events.insert({
            calendarId: calendarID,
            resource: event,
        });

        // Generar la URL de la invitación
        const eventId = response.data.id;
        console.log('Evento creado con éxito');
        return eventId;
    } catch (err) {
        console.error('Hubo un error al crear el evento en el servicio de Calendar:', err);
        throw err;
    }
}


// Función para obtener los eventos dentro del rango especificado por dateLimit
/**
 * Obtiene los próximos eventos dentro del rango especificado por dateLimit.
 * @param {number} [dateLimit=dateLimit] - Máximo de días en adelante para buscar eventos. Default es el valor de la constante dateLimit.
 * @returns {Array} - Lista de eventos dentro del rango especificado.
 */
async function getNextEvents(dateLimit = dateLimit) {
    try {
        // Autenticación
        const authClient = await auth.getClient();
        google.options({ auth: authClient });

        // Fecha de inicio (ahora)
        const now = new Date();
        // Fecha de fin (ahora + dateLimit días)
        const endDate = new Date(now);
        endDate.setDate(now.getDate() + dateLimit);

        const response = await calendar.events.list({
            calendarId: calendarID,
            timeMin: now.toISOString(),
            timeMax: endDate.toISOString(),
            timeZone: timeZone,
            singleEvents: true,
            orderBy: 'startTime'
        });

        const events = response.data.items;
        if (events.length) {
            events.forEach((event, i) => {
                const start = event.start.dateTime || event.start.date;
            });
        }
        return events;
    } catch (err) {
        console.error('Hubo un error al contactar el servicio de Calendar: ' + err);
        throw err;
    }
}

/**
 * Lista los slots disponibles entre las fechas dadas.
 * @param {Date} [startDate=new Date()] - Fecha de inicio para buscar slots disponibles. Default es la fecha actual.
 * @param {Date} [endDate] - Fecha de fin para buscar slots disponibles. Default es el máximo definido por dateLimit.
 * @returns {Array} - Lista de slots disponibles.
 */
async function listAvailableSlots(startDate = new Date(), endDate) {
    try {
        const authClient = await auth.getClient();
        google.options({ auth: authClient });

        // Definir fecha de fin si no se proporciona
        if (!endDate) {
            endDate = new Date(startDate);
            endDate.setDate(startDate.getDate() + dateLimit);
        }

        const response = await calendar.events.list({
            calendarId: calendarID,
            timeMin: startDate.toISOString(),
            timeMax: endDate.toISOString(),
            timeZone: timeZone,
            singleEvents: true,
            orderBy: 'startTime'
        });

        const events = response.data.items;
        const slots = [];
        let currentDate = new Date(startDate);

        // Generar slots disponibles basados en rangeLimit
        while (currentDate < endDate) {
            const dayOfWeek = currentDate.getDay();
            if (rangeLimit.days.includes(dayOfWeek)) {
                for (let hour = rangeLimit.startHour; hour < rangeLimit.endHour; hour++) {
                    const slotStart = new Date(currentDate);
                    slotStart.setHours(hour, 0, 0, 0);
                    const slotEnd = new Date(slotStart);
                    slotEnd.setHours(hour + standardDuration);

                    const isBusy = events.some(event => {
                        const eventStart = new Date(event.start.dateTime || event.start.date);
                        const eventEnd = new Date(event.end.dateTime || event.end.date);
                        return (slotStart < eventEnd && slotEnd > eventStart);
                    });

                    if (!isBusy) {
                        slots.push({ start: slotStart, end: slotEnd });
                    }
                }
            }
            currentDate.setDate(currentDate.getDate() + 1);
        }

        return slots;
    } catch (err) {
        console.error('Hubo un error al contactar el servicio de Calendar: ' + err);
        throw err;
    }
}

/**
 * Verifica si hay slots disponibles para una fecha dada.
 * @param {Date} date - Fecha a verificar.
 * @returns {boolean} - Devuelve true si hay slots disponibles dentro del rango permitido, false en caso contrario.
 */
async function isDateAvailable(date) {
    try {
        // Validar que la fecha esté dentro del rango permitido
        const currentDate = new Date();
        const maxDate = new Date(currentDate);
        maxDate.setDate(currentDate.getDate() + dateLimit);

        if (date < currentDate || date > maxDate) {
            return false; // La fecha está fuera del rango permitido
        }

        // Verificar que la fecha caiga en un día permitido
        const dayOfWeek = date.getDay();
        if (!rangeLimit.days.includes(dayOfWeek)) {
            return false; // La fecha no está dentro de los días permitidos
        }

        // Verificar que la hora esté dentro del rango permitido
        const hour = date.getHours();
        if (hour < rangeLimit.startHour || hour >= rangeLimit.endHour) {
            return false; // La hora no está dentro del rango permitido
        }

        // Obtener todos los slots disponibles desde la fecha actual hasta el límite definido
        const availableSlots = await listAvailableSlots(currentDate);

        // Filtrar slots disponibles basados en la fecha dada
        const slotsOnGivenDate = availableSlots.filter(slot => new Date(slot.start).toDateString() === date.toDateString());

        // Verificar si hay slots disponibles en la fecha dada
        const isSlotAvailable = slotsOnGivenDate.some(slot =>
            new Date(slot.start).getTime() === date.getTime() &&
            new Date(slot.end).getTime() === date.getTime() + standardDuration * 60 * 60 * 1000
        );

        return isSlotAvailable;
    } catch (err) {
        console.error('Hubo un error al verificar disponibilidad de la fecha: ' + err);
        throw err;
    }
}


/**
 * Obtiene el próximo slot disponible a partir de la fecha dada.
 * @param {string|Date} date - Fecha a partir de la cual buscar el próximo slot disponible, puede ser un string en formato ISO o un objeto Date.
 * @returns {Object|null} - El próximo slot disponible o null si no hay ninguno.
 */
async function getNextAvailableSlot(date) {
    try {
        // Verificar si 'date' es un string en formato ISO
        if (typeof date === 'string') {
            // Convertir el string ISO en un objeto Date
            date = new Date(date);
        } else if (!(date instanceof Date) || isNaN(date)) {
            throw new Error('La fecha proporcionada no es válida.');
        }

        // Obtener el próximo slot disponible
        const availableSlots = await listAvailableSlots(date);

        // Filtrar slots disponibles que comienzan después de la fecha proporcionada
        const filteredSlots = availableSlots.filter(slot => new Date(slot.start) > date);

        // Ordenar los slots por su hora de inicio en orden ascendente
        const sortedSlots = filteredSlots.sort((a, b) => new Date(a.start) - new Date(b.start));

        // Tomar el primer slot de la lista resultante, que será el próximo slot disponible más cercano
        return sortedSlots.length > 0 ? sortedSlots[0] : null;
    } catch (err) {
        console.error('Hubo un error al obtener el próximo slot disponible: ' + err);
        throw err;
    }
}


module.exports = { createEvent, isDateAvailable, getNextAvailableSlot };