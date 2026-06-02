import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

void main() => runApp(const MyApp());

const String baseUrl = 'https://skeletal-serrated-blinker.ngrok-free.dev';
const Map<String, String> commonHeaders = {
  "ngrok-skip-browser-warning": "69420",
  "Content-Type": "application/json",
};

const Color pointBlue = Color(0xFF1A73E8);

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primaryColor: pointBlue,
        colorScheme: ColorScheme.fromSeed(seedColor: pointBlue),
        useMaterial3: true,
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: Colors.grey[100],
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(8), borderSide: BorderSide.none),
          contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: pointBlue,
            foregroundColor: Colors.white,
            minimumSize: const Size(double.infinity, 50),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
            elevation: 0,
          ),
        ),
      ),
      home: const LoginScreen(),
    );
  }
}

String getBiasText(int? score) {
  if (score == 0) return "진보";
  if (score == 1) return "중도";
  if (score == 2) return "보수";
  return "확인불가";
}

// --- 회원가입 ---
class SignUpScreen extends StatefulWidget {
  const SignUpScreen({super.key});
  @override
  State<SignUpScreen> createState() => _SignUpScreenState();
}

class _SignUpScreenState extends State<SignUpScreen> {
  final TextEditingController _id = TextEditingController();
  final TextEditingController _pw = TextEditingController();
  bool _isIdChecked = false;

  Future<void> _checkId() async {
    if (_id.text.isEmpty) return;
    try {
      final res = await http.get(Uri.parse('$baseUrl/check_id/${_id.text}'), headers: commonHeaders);
      final data = jsonDecode(res.body);
      if (data['available'] == true) {
        setState(() => _isIdChecked = true);
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("사용 가능한 아이디입니다."), backgroundColor: Colors.green));
      } else {
        setState(() => _isIdChecked = false);
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(data['message']), backgroundColor: Colors.orange));
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("서버 연결 실패")));
    }
  }

  Future<void> _reg() async {
    if (!_isIdChecked) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("아이디 중복 확인을 해주세요."), backgroundColor: Colors.redAccent));
      return;
    }
    try {
      await http.post(Uri.parse('$baseUrl/register'), headers: commonHeaders, body: jsonEncode({"id": _id.text, "password": _pw.text}));
      if (!mounted) return;
      Navigator.pop(context);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("서버 연결 실패")));
    }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(centerTitle: true, leading: const BackButton()),
        body: SingleChildScrollView(
          padding: const EdgeInsets.all(30),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              const Icon(Icons.account_circle, size: 60, color: pointBlue),
              const SizedBox(height: 10),
              const Text("회원가입", style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
              const Text("계정을 만들고 뉴스 분석 서비스를 이용해 보세요", style: TextStyle(color: Colors.grey)),
              const SizedBox(height: 40),
              Row(
                children: [
                  Expanded(child: TextField(controller: _id, decoration: const InputDecoration(hintText: "아이디를 입력하세요"), onChanged: (v) => setState(() => _isIdChecked = false))),
                  const SizedBox(width: 10),
                  ElevatedButton(onPressed: _checkId, style: ElevatedButton.styleFrom(minimumSize: const Size(100, 50)), child: const Text("중복 확인")),
                ],
              ),
              const SizedBox(height: 15),
              TextField(controller: _pw, decoration: const InputDecoration(hintText: "비밀번호를 입력하세요"), obscureText: true),
              const SizedBox(height: 30),
              ElevatedButton(onPressed: _reg, child: const Text("회원가입")),
            ],
          ),
        ),
      );
}

// --- 로그인 ---
class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});
  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final TextEditingController _id = TextEditingController();
  final TextEditingController _pw = TextEditingController();
  bool _doAutoLogin = false;

  @override
  void initState() {
    super.initState();
    _checkSavedLogin();
  }

  void _checkSavedLogin() async {
    final prefs = await SharedPreferences.getInstance();
    if (prefs.getBool('isLoggedIn') ?? false) {
      if (!mounted) return;
      Navigator.pushReplacement(context, MaterialPageRoute(builder: (c) => const HomeScreen(isGuest: false)));
    }
  }

  void _login(bool guest) async {
    if (guest) {
  final prefs = await SharedPreferences.getInstance();
  await prefs.remove('user_idx');
  await prefs.remove('user_id');
  await prefs.setBool('isLoggedIn', false);

  if (!mounted) return;
  Navigator.pushReplacement(
    context,
    MaterialPageRoute(builder: (c) => const HomeScreen(isGuest: true)),
  );
  return;
}
    try {
      final res = await http.post(Uri.parse('$baseUrl/login'), headers: commonHeaders, body: jsonEncode({"id": _id.text, "password": _pw.text}));
      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        final prefs = await SharedPreferences.getInstance();
        await prefs.setInt('user_idx', data['user_idx']);
        await prefs.setString('user_id', _id.text); 
        await prefs.setBool('isLoggedIn', _doAutoLogin);
        if (!mounted) return;
        Navigator.pushReplacement(context, MaterialPageRoute(builder: (c) => const HomeScreen(isGuest: false)));
      } else {
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("아이디 또는 비밀번호가 틀렸습니다."), backgroundColor: Colors.redAccent));
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("서버 연결 실패")));
    }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        body: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(30),
            child: Column(
              children: [
                const Icon(Icons.verified_user, size: 60, color: pointBlue),
                const SizedBox(height: 10),
                const Text("N 앱 이름", style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold, color: pointBlue)),
                const Text("로그인하고 다양한 뉴스 분석 결과를 확인해 보세요", style: TextStyle(color: Colors.grey)),
                const SizedBox(height: 40),
                TextField(controller: _id, decoration: const InputDecoration(hintText: "아이디를 입력하세요")),
                const SizedBox(height: 15),
                TextField(controller: _pw, decoration: const InputDecoration(hintText: "비밀번호를 입력하세요"), obscureText: true),
                Row(
                  children: [
                    Checkbox(value: _doAutoLogin, onChanged: (v) => setState(() => _doAutoLogin = v!), activeColor: pointBlue),
                    const Text("자동 로그인"),
                  ],
                ),
                const SizedBox(height: 20),
                ElevatedButton(onPressed: () => _login(false), child: const Text("로그인")),
                const SizedBox(height: 10),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Text("아직 계정이 없으신가요?"),
                    TextButton(onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (c) => const SignUpScreen())), child: const Text("회원가입", style: TextStyle(color: pointBlue))),
                  ],
                ),
                TextButton(onPressed: () => _login(true), child: const Text("로그인 없이 시작하기", style: TextStyle(color: Colors.grey)))
              ],
            ),
          ),
        ),
      );
}

// --- 메인 홈 ---
class HomeScreen extends StatefulWidget {
  final bool isGuest;
  const HomeScreen({super.key, required this.isGuest});
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  List<dynamic> news = [];
  String cat = '전체';
  int _selectedIndex = 0;
  final TextEditingController _s = TextEditingController();

  Future<void> load({String? c, String? q}) async {
    String url = '$baseUrl/news?';
    if (c != null && c != '전체') url += 'category=$c&';
    if (q != null) url += 'search=$q';
    final res = await http.get(Uri.parse(url), headers: commonHeaders);
    if (res.statusCode == 200) {
      setState(() {
        news = jsonDecode(utf8.decode(res.bodyBytes));
      });
    }
  }

  @override
  void initState() {
    super.initState();
    load();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(
          title: Container(
            height: 40,
            decoration: BoxDecoration(color: Colors.grey[200], borderRadius: BorderRadius.circular(20)),
            child: TextField(
              controller: _s,
              decoration: const InputDecoration(hintText: "뉴스 검색...", prefixIcon: Icon(Icons.search), border: InputBorder.none, contentPadding: EdgeInsets.zero),
              onSubmitted: (v) => load(q: v),
            ),
          ),
          actions: [IconButton(icon: const Icon(Icons.notifications_none), onPressed: () {})],
        ),
        body: Column(
          children: [
            SingleChildScrollView(
                scrollDirection: Axis.horizontal,
                padding: const EdgeInsets.symmetric(vertical: 10),
                child: Row(
                    children: ['전체', '정치', '경제', '사회', '생활/문화', 'IT/과학', '엔터', '스포츠']
                        .map((category) => Padding(
                            padding: const EdgeInsets.symmetric(horizontal: 4),
                            child: ChoiceChip(
                                label: Text(category),
                                selected: cat == category,
                                selectedColor: pointBlue,
                                labelStyle: TextStyle(color: cat == category ? Colors.white : Colors.black),
                                onSelected: (s) {
                                  setState(() => cat = category);
                                  load(c: category);
                                })))
                        .toList())),
            Expanded(
                child: news.isEmpty
                    ? const Center(child: Text("표시할 뉴스가 없습니다."))
                    : ListView.builder(
                        itemCount: news.length,
                        itemBuilder: (c, i) => Card(
                              margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                              elevation: 0,
                              clipBehavior: Clip.antiAlias, 
                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12), side: BorderSide(color: Colors.grey[200]!)),
                              child: InkWell(
                                onTap: () => Navigator.push(context, MaterialPageRoute(builder: (c) => DetailScreen(idx: news[i]['idx'], isGuest: widget.isGuest))),
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    if (news[i]['picture'] != null && news[i]['picture'].toString().isNotEmpty)
                                      Image.network(
                                        news[i]['picture'],
                                        headers: commonHeaders, // 💡 헤더 추가 완료
                                        width: double.infinity,
                                        height: 180,
                                        fit: BoxFit.cover,
                                        errorBuilder: (context, error, stackTrace) => Container(height: 180, color: Colors.grey[300], child: const Icon(Icons.image, color: Colors.grey)),
                                      )
                                    else
                                      Container(height: 180, width: double.infinity, color: Colors.grey[300], child: const Icon(Icons.image, color: Colors.grey, size: 50)),
                                    
                                    Padding(
                                      padding: const EdgeInsets.all(16),
                                      child: Column(
                                        crossAxisAlignment: CrossAxisAlignment.start,
                                        children: [
                                          Wrap(
                                            crossAxisAlignment: WrapCrossAlignment.center,
                                            children: [
                                              Text(news[i]['title'], style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
                                              if (news[i]['clickbait_score'] == 1) ...[
                                                const SizedBox(width: 6),
                                                Container(
                                                  padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                                                  decoration: BoxDecoration(color: Colors.orange[50], borderRadius: BorderRadius.circular(4)),
                                                  child: const Text("⚠️ 낚시주의", style: TextStyle(color: Colors.orange, fontSize: 10, fontWeight: FontWeight.bold)),
                                                ),
                                              ]
                                            ],
                                          ),
                                          const SizedBox(height: 8),
                                          Row(children: [
                                            if (news[i]['category'] == '정치')
                                              Container(
                                                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                                                decoration: BoxDecoration(color: Colors.red[50], borderRadius: BorderRadius.circular(4)),
                                                child: Text("⚖️ ${getBiasText(news[i]['bias_score'])}", style: const TextStyle(color: Colors.red, fontSize: 12)),
                                              ),
                                            const Spacer(),
                                            Text(news[i]['category'], style: const TextStyle(fontSize: 12, color: Colors.grey)),
                                          ]),
                                        ],
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ))),
          ],
        ),
        bottomNavigationBar: BottomNavigationBar(
          currentIndex: _selectedIndex,
          selectedItemColor: pointBlue,
          onTap: (index) async {
            if (index == 1) { // 💡 추천 페이지 연결
              await Navigator.push(context, MaterialPageRoute(builder: (c) => RecommendScreen(isGuest: widget.isGuest)));
              if (mounted) setState(() => _selectedIndex = 0);
            } else if (index == 2) {
              await Navigator.push(context, MaterialPageRoute(builder: (c) => MyPageScreen(isGuest: widget.isGuest),),);
              if (mounted) setState(() => _selectedIndex = 0);
            } else {
              setState(() => _selectedIndex = index);
            }
          },
          items: const [
            BottomNavigationBarItem(icon: Icon(Icons.home), label: "홈"),
            BottomNavigationBarItem(icon: Icon(Icons.thumb_up), label: "추천"),
            BottomNavigationBarItem(icon: Icon(Icons.person), label: "마이페이지"),
          ],
        ),
      );
}

// --- 💡 [신규 추가] 추천 페이지 (RecommendScreen) ---
class RecommendScreen extends StatefulWidget {
  final bool isGuest;
  const RecommendScreen({super.key, required this.isGuest});
  @override
  State<RecommendScreen> createState() => _RecommendScreenState();
}

class _RecommendScreenState extends State<RecommendScreen> {
  List<dynamic> recommendNews = [];
  bool isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadRecommendations();
  }

  void _loadRecommendations() async {
    if (widget.isGuest) {
      setState(() => isLoading = false);
      return;
    }

    final prefs = await SharedPreferences.getInstance();
    final userIdx = prefs.getInt('user_idx');

    try {
      // 💡 백엔드 추천 API 주소 (필요에 따라 수정)
      final res = await http.get(Uri.parse('$baseUrl/recommendations/$userIdx'), headers: commonHeaders);
      if (res.statusCode == 200) {
        setState(() {
          recommendNews = jsonDecode(utf8.decode(res.bodyBytes));
          isLoading = false;
        });
      } else {
        setState(() => isLoading = false);
      }
    } catch (e) {
      setState(() => isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("맞춤 뉴스 추천"), centerTitle: true),
      body: widget.isGuest
          ? Center(
              child: ElevatedButton(
                onPressed: () => Navigator.pushAndRemoveUntil(context, MaterialPageRoute(builder: (c) => const LoginScreen()), (r) => false),
                style: ElevatedButton.styleFrom(minimumSize: const Size(200, 50)),
                child: const Text("로그인하고 추천 받기"),
              ),
            )
          : isLoading
              ? const Center(child: CircularProgressIndicator())
              : recommendNews.isEmpty
                  ? const Center(child: Text("아직 추천 뉴스가 없습니다. 좋아요를 눌러보세요!"))
                  : ListView.builder(
                      itemCount: recommendNews.length,
                      itemBuilder: (c, i) => ListTile(
                        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                        leading: ClipRRect(
                          borderRadius: BorderRadius.circular(8),
                          child: recommendNews[i]['picture'] != null && recommendNews[i]['picture'].toString().isNotEmpty
                              ? Image.network(
                                  recommendNews[i]['picture'], 
                                  headers: commonHeaders, // 💡 헤더 추가 완료
                                  width: 80, height: 80, fit: BoxFit.cover,
                                  errorBuilder: (context, error, stackTrace) => Container(width: 80, height: 80, color: Colors.grey[300], child: const Icon(Icons.image, color: Colors.grey)))
                              : Container(width: 80, height: 80, color: Colors.grey[300], child: const Icon(Icons.image, color: Colors.grey)),
                        ),
                        title: Text(recommendNews[i]['title'], maxLines: 2, overflow: TextOverflow.ellipsis),
                        subtitle: Text(recommendNews[i]['category'] ?? ''),
                        onTap: () => Navigator.push(context, MaterialPageRoute(builder: (c) => DetailScreen(idx: recommendNews[i]['idx'], isGuest: false))),
                      ),
                    ),
    );
  }
}

// --- 마이페이지 ---
class MyPageScreen extends StatefulWidget {
  final bool isGuest;

  const MyPageScreen({super.key, required this.isGuest});

  @override
  State<MyPageScreen> createState() => _MyPageScreenState();
}

class _MyPageScreenState extends State<MyPageScreen> {
  List<dynamic> likes = [];
  bool isGuest = true;
  String? userId;

  void load() async {
  if (widget.isGuest) {
    setState(() {
      isGuest = true;
      userId = null;
      likes = [];
    });
    return;
  }

  final prefs = await SharedPreferences.getInstance();
  final userIdx = prefs.getInt('user_idx');
  final savedId = prefs.getString('user_id');

  if (userIdx != null) {
    setState(() {
      isGuest = false;
      userId = savedId;
    });

    final res = await http.get(
      Uri.parse('$baseUrl/user/$userIdx/likes'),
      headers: commonHeaders,
    );

    if (res.statusCode == 200) {
      setState(() {
        likes = jsonDecode(utf8.decode(res.bodyBytes));
      });
    }
  } else {
    setState(() {
      isGuest = true;
      userId = null;
      likes = [];
    });
  }
}

  void _logout() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.clear();
    if (!mounted) return;
    Navigator.pushAndRemoveUntil(context, MaterialPageRoute(builder: (c) => const LoginScreen()), (r) => false);
  }

  @override
  void initState() {
    super.initState();
    load();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: const Text("내가 좋아요 한 뉴스"), centerTitle: true),
        body: isGuest
            ? Center(
                child: ElevatedButton(
                  onPressed: () => Navigator.pushAndRemoveUntil(context, MaterialPageRoute(builder: (c) => const LoginScreen()), (r) => false),
                  style: ElevatedButton.styleFrom(minimumSize: const Size(200, 50)),
                  child: const Text("로그인 하러 가기"),
                ),
              )
            : Column(
                children: [
                  ListTile(
                    leading: const CircleAvatar(backgroundColor: pointBlue, child: Icon(Icons.person, color: Colors.white)), 
                    title: const Text("회원님"), 
                    subtitle: Text(userId ?? "알 수 없음")
                  ),
                  const Divider(),
                  Padding(
                    padding: const EdgeInsets.all(16), 
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        const Text("좋아요 한 뉴스", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
                        TextButton(
                          onPressed: _logout, 
                          child: const Text("로그아웃", style: TextStyle(color: Colors.red, fontWeight: FontWeight.bold))
                        )
                      ],
                    )
                  ),
                  Expanded(
                    child: likes.isEmpty
                        ? const Center(child: Text("좋아요 한 뉴스가 없습니다."))
                        : ListView.builder(
                            itemCount: likes.length,
                            itemBuilder: (c, i) => ListTile(
                                  contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                                  leading: ClipRRect(
                                    borderRadius: BorderRadius.circular(8),
                                    child: likes[i]['picture'] != null && likes[i]['picture'].toString().isNotEmpty
                                        ? Image.network(
                                            likes[i]['picture'], 
                                            headers: commonHeaders, // 💡 헤더 추가 완료
                                            width: 80, height: 80, fit: BoxFit.cover, 
                                            errorBuilder: (context, error, stackTrace) => Container(width: 80, height: 80, color: Colors.grey[300], child: const Icon(Icons.image, color: Colors.grey)))
                                        : Container(width: 80, height: 80, color: Colors.grey[300], child: const Icon(Icons.image, color: Colors.grey)),
                                  ),
                                  title: Text(likes[i]['title'], maxLines: 2, overflow: TextOverflow.ellipsis),
                                  onTap: () => Navigator.push(context, MaterialPageRoute(builder: (c) => DetailScreen(idx: likes[i]['idx'], isGuest: false))),
                                )),
                  ),
                ],
              ),
      );
}

// --- 상세 페이지 ---
class DetailScreen extends StatefulWidget {
  final int idx;
  final bool isGuest;
  const DetailScreen({super.key, required this.idx, required this.isGuest});
  @override
  State<DetailScreen> createState() => _DetailScreenState();
}

class _DetailScreenState extends State<DetailScreen> {
  Map<String, dynamic>? data;
  bool isLoading = true;

  void load() async {
    try {
      final res = await http.get(
        Uri.parse('$baseUrl/news/${widget.idx}'),
        headers: commonHeaders,
      );
      if (res.statusCode == 200) {
        setState(() { 
          data = jsonDecode(utf8.decode(res.bodyBytes)); 
          isLoading = false;
        });
      } else {
        setState(() => isLoading = false);
      }
    } catch (e) {
      setState(() => isLoading = false);
    }
  }

  void like() async {
    final prefs = await SharedPreferences.getInstance();
    await http.post(
      Uri.parse('$baseUrl/like'), 
      headers: commonHeaders,
      body: jsonEncode({"user_idx": prefs.getInt('user_idx'), "article_idx": widget.idx})
    );
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("좋아요 목록에 추가되었습니다!")));
  }

  @override
  void initState() { super.initState(); load(); }

  @override
  Widget build(BuildContext context) {
    if (isLoading) return const Scaffold(body: Center(child: CircularProgressIndicator()));
    if (data == null) return const Scaffold(body: Center(child: Text("데이터를 불러올 수 없습니다.")));

    String? factCheckText;
    if (data!['fact_check_results'] != null) {
      factCheckText = data!['fact_check_results'].toString();
    }

    return Scaffold(
      appBar: AppBar(
        title: Text(data!['category'] ?? "상세 분석"),
        centerTitle: true,
        actions: [if (!widget.isGuest) IconButton(icon: const Icon(Icons.bookmark_border), onPressed: like)]
      ),
      body: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start, 
          children: [
            if (data!['picture'] != null && data!['picture'].toString().isNotEmpty)
              Image.network(
                data!['picture'],
                headers: commonHeaders, // 💡 헤더 추가 완료
                width: double.infinity,
                height: 250,
                fit: BoxFit.cover,
                errorBuilder: (context, error, stackTrace) => Container(height: 250, width: double.infinity, color: Colors.grey[300], child: const Icon(Icons.image, color: Colors.grey, size: 50)),
              )
            else
              Container(height: 250, width: double.infinity, color: Colors.grey[300], child: const Icon(Icons.image, color: Colors.grey, size: 50)),
            
            Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start, 
                children: [
                  Text(data!['title'] ?? "제목 없음", style: const TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
                  const SizedBox(height: 10),
                  Row(
                    children: [
                      Text("${data!['category'] ?? '분류 없음'} | 24시간 전", style: const TextStyle(color: Colors.grey)),
                      if (data!['category'] == '정치') ...[
                        const SizedBox(width: 10),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                          decoration: BoxDecoration(color: Colors.red[50], borderRadius: BorderRadius.circular(4)),
                          child: Text(
                            "⚖️ 편향: ${getBiasText(data!['bias_score'])}", 
                            style: const TextStyle(color: Colors.red, fontSize: 12, fontWeight: FontWeight.bold)
                          ),
                        ),
                      ],
                      if ((data!['category'] == '엔터' || data!['category'] == '스포츠') && data!['clickbait_score'] == 1) ...[
                        const SizedBox(width: 10),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                          decoration: BoxDecoration(color: Colors.orange[50], borderRadius: BorderRadius.circular(4)),
                          child: const Text(
                            "⚠️ 낚시 주의", 
                            style: TextStyle(color: Colors.orange, fontSize: 12, fontWeight: FontWeight.bold)
                          ),
                        ),
                      ],
                    ],
                  ),
                  const Divider(height: 40),
                  Text(
                    data!['content']?.toString().isNotEmpty == true ? data!['content'] : "본문 내용을 불러올 수 없는 기사입니다.", 
                    style: const TextStyle(fontSize: 16, height: 1.6)
                  ),
                  const SizedBox(height: 40),
                  
                  if (factCheckText != null && factCheckText.isNotEmpty) ...[
                    const Row(
                      children: [
                        Icon(Icons.fact_check, color: pointBlue), 
                        SizedBox(width: 8), 
                        Text("AI 팩트체크 분석", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: pointBlue))
                      ]
                    ),
                    const SizedBox(height: 15),
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: Colors.grey[50], 
                        borderRadius: BorderRadius.circular(12), 
                        border: Border.all(color: Colors.grey[200]!)
                      ),
                      child: Text(
                        factCheckText,
                        style: const TextStyle(fontSize: 15, height: 1.6, color: Colors.black87),
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ]
        )
      ),
    );
  }
}
// 실행코드 C:\flutter\bin\flutter.bat build apk --release