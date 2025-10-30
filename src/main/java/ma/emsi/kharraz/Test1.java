package ma.emsi.kharraz;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;

public class Test1 {

    private static final String GEMINI_KEY_ENV = "GEMINI_KEY";

    public static void main(String[] args) {
        // 1. Récupérer la clé d'API depuis les variables d'environnement
        String apiKey = System.getenv(GEMINI_KEY_ENV);
        if (apiKey == null || apiKey.isBlank()) {
            System.err.println("La clé d'API Google Gemini est introuvable. Définissez la variable d'environnement 'GEMINI_KEY'.");
            return;
        }

        // 2. Créer et configurer le modèle Gemini
        ChatModel geminiModel = buildGeminiModel(apiKey);

        // 3. Préparer la requête à envoyer au modèle
        String question = "Quel est la ville que les touristes visitent le plus en France?";

        // 4. Interroger le modèle et récupérer la réponse
        String response = geminiModel.chat(question);

        // 5. Afficher la réponse
        System.out.println("Réponse du modèle : " + response);
    }

    private static ChatModel buildGeminiModel(String apiKey) {
        return GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .build();
    }
}
