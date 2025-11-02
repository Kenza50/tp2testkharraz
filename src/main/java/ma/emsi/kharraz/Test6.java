package ma.emsi.kharraz;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.service.AiServices;

import java.util.Scanner;

public class Test6 {
    public static void main(String[] args) {
        // Crée un ChatModel Gemini avec les logs activés
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(System.getenv("GEMINI_KEY"))
                .modelName("gemini-2.5-flash")
                .logRequests(true)
                .logResponses(true)
                .build();

        // Crée un assistant météo avec l'outil MeteoTool
        AssistantMeteo assistant = AiServices.builder(AssistantMeteo.class)
                .chatModel(model)
                .tools(new MeteoTool())
                .build();

        // Démarre une boucle interactive pour discuter avec l'assistant
        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("Vous: ");
            String query = scanner.nextLine();

            if (query.equalsIgnoreCase("exit")) {
                break;
            }

            String response = assistant.chat(query);
            System.out.println("Assistant: " + response);
        }
        scanner.close();
    }
}