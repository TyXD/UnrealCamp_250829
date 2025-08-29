# [내일배움캠프 언리얼5기] 그래프, 백트래킹

## 학습 키워드

>## 그래프

>## 백트래킹

## 학습 내용

## 그래프

그래프는 유한한 수의 정점과 이들을 연결하는 간선들의 집합으로 구성된 비선형 자료구조입니다.

보통 데이터의 관계를 표현하는 데 사용되며 데이터는 노드로,

데이터 간의 관계나 흐름은 간선으로 표현됩니다.

### 그래프의 구성 요소

>### 노드

데이터를 의미하고, 보통 동그라미로 표현합니다.

>### 간선

노드와 노드를 연결하는 선입니다..

>### 가중치

간선에 숫자를 추가하여 표현하면 가중치입니다.

### 그래프의 특징

그래프는 방향성, 가중치, 순환 특성에 따라 종류를 구분합니다.

간선의 방향의 유무에 따라 방향 그래프, 무방향 그래프로 나눕니다.

간선에 가중치가 있는 그래프를 가중치 그래프라 합니다.

시작 노드에서 다시 시작 노드로 돌아오는 경로가 있으면 순환 그래프라 합니다.

순환 경로가 존재하지 않는 그래프를 비순환 그래프라 합니다.

### 그래프의 구현 방식

>### 인접 행렬

배열을 활용해서 구현하는 방식을 인접 행렬이라 합니다.

배열의 인덱스는 노드이고, 배열의 값은 가중치입니다.

```cpp

// 목적: 인접 행렬을 사용해 가중치가 있는 방향 그래프를 구현
// 동작: 서울(0)에서 부산(1)로 가는 간선이 있으며, 가중치는 400

#include <iostream>
using namespace std;

int main() {
    const int N = 2; // 노드 수 (서울, 부산)
    int graph[N][N] = {0}; // 초기화된 인접 행렬

    graph[0][1] = 400; // 서울(0) -> 부산(1)

    cout << "서울 → 부산 가중치: " << graph[0][1] << endl;

    return 0;
}

/*
출력결과
서울 → 부산 가중치: 400
*/

```
		- 서울 부산 노드가 있는 가중치 그래프

인접 행렬은 특정 간선이 있는지 확인하는 연산 속도가 매우 빠릅니다.(O(1))

또한 인접 행렬로 구현하는 코드의 난이도가 낮습니다.

다만 노드 수에 비해 간선의 수가 적은 그래프의 경우 할당되는 메모리의 낭비가 심합니다.

이는 인접행렬의 경우 간선의 수와 관계없이 무조건 노드의 수 V X V 크기의 인접 행렬이 만들어지기 때문입니다.

```cpp

// 목적: 인접 행렬을 통해 간선 존재 여부를 O(1)로 확인하는 장점 설명
// 동작: 노드 0 → 1 간선이 있는지 빠르게 확인

#include <iostream>
using namespace std;

int main() {
    const int V = 3;
    int graph[V][V] = {0};

    graph[0][1] = 1; // 0 → 1 간선 추가

    // 특정 간선 존재 여부 확인 (O(1))
    if (graph[0][1]) {
        cout << "간선 존재\n";
    } else {
        cout << "간선 없음\n";
    }

    return 0;
}

/*
출력결과
간선 존재
*/
```
		- 인접 행렬에서 간선이 있는지 파악하는 속도가 빠른 이유

```cpp

 // 목적: 인접 행렬의 메모리 낭비 문제를 설명
// 동작: 노드는 1000개지만 간선은 2개만 있는 희소 그래프를 인접 행렬로 표현

#include <iostream>
using namespace std;

int main() {
    const int V = 1000;
    int graph[V][V] = {0}; // 약 4MB 이상 메모리 소모

    // 실제 간선은 단 2개뿐
    graph[1][2] = 1;
    graph[5][999] = 1;

    cout << "1 → 2: " << graph[1][2] << endl;
    cout << "5 → 999: " << graph[5][999] << endl;

    return 0;
}

/*
출력결과
1 → 2: 1
5 → 999: 1

※ 대부분의 graph[i][j]는 0이므로 메모리 낭비 심함
*/
```
		- 간선의 수가 적어도 무조건 VxV만큼의 메모리를 할당해야한다.


>### 인접 리스트

인접 리스트는 노드와 간선을 표현하는 구조체와 이들을 자료형으로 가지는 배열들로 이루어져 있습니다.

구조체는 각각 연결된 노드와 해당 간선의 가중치를 가지고 있으며, 배열의 원소는 노드를 의미합니다.

즉 배열의 첫원째 원소는 첫번째 노드이며, 첫번째 배열에 들어있는 구조체의 갯수만큼 간선을 가지고 있습니다.

```cpp

// 목적: 숫자 노드 기반으로 여러 간선이 있는 인접 리스트 구현
// 동작: 노드 0~3까지 여러 간선을 설정하여 출력

#include <iostream>
#include <vector>
using namespace std;

struct Node {
    int v;
    int w;
};

int main() {
    const int N = 4;
    vector<Node> graph[N];

    graph[0].push_back({1, 5});
    graph[0].push_back({2, 7});
    graph[1].push_back({2, 3});
    graph[2].push_back({3, 8});

    for (int i = 0; i < N; ++i) {
        for (const auto& node : graph[i]) {
            cout << i << " → " << node.v << " (가중치: " << node.w << ")\n";
        }
    }

    return 0;
}

/*
출력결과
0 → 1 (가중치: 5)
0 → 2 (가중치: 7)
1 → 2 (가중치: 3)
2 → 3 (가중치: 8)
*/

```
		- v는 연결되어 있는 노드 w는 가중치, N은 노드의 총 개수

인접 리스트는 실제로 연결된 노드만 메모리에 할당되므로

메모리를 인접 행렬과 비교하면 매우 절약할 수 있습니다.

다만 특정 간선을 확인하는데 걸리는 시간이 매우 오래 걸립니다.(O(N))

```cpp

// 목적: 인접 리스트에서 특정 간선 존재 여부 확인 시 O(E) 시간이 걸릴 수 있는 단점 설명
// 동작: 0번 노드에서 999번 노드로 가는 간선이 있는지 선형 탐색

#include <iostream>
#include <vector>
using namespace std;

struct Node {
    int v, w;
};

int main() {
    const int N = 1000;
    vector<Node> graph[N];

    // 0번 노드에서 1~998까지 연결 (간선 많음)
    for (int i = 1; i < 999; ++i) {
        graph[0].push_back({i, i});
    }

    // 마지막에 999 연결
    graph[0].push_back({999, 999});

    // 특정 간선(0 → 999) 존재 여부 확인: 선형 탐색 (O(E))
    bool found = false;
    for (const auto& node : graph[0]) {
        if (node.v == 999) {
            found = true;
            break;
        }
    }

    cout << (found ? "간선 존재" : "간선 없음") << endl;

    return 0;
}

/*
출력결과
간선 존재

※ 0 → 999 간선을 찾기 위해 최대 999번 비교 필요할 수 있음 (O(E))
*/
```
		- 특정 간선을 찾는데 걸리는 시간

```cpp

// 목적: 인접 리스트는 연결된 노드만 저장하여 메모리를 절약할 수 있는 장점 설명
// 동작: 노드 1000개 중 단 2개만 연결된 희소 그래프를 인접 리스트로 구현

#include <iostream>
#include <vector>
using namespace std;

struct Node {
    int v, w;
};

int main() {
    const int N = 1000;
    vector<Node> graph[N];

    graph[1].push_back({2, 5});
    graph[5].push_back({999, 10});

    cout << "1 → " << graph[1][0].v << " (가중치: " << graph[1][0].w << ")\n";
    cout << "5 → " << graph[5][0].v << " (가중치: " << graph[5][0].w << ")\n";

    return 0;
}

/*
출력결과
1 → 2 (가중치: 5)
5 → 999 (가중치: 10)

※ 연결된 간선만 저장하므로 메모리 낭비 없음
*/
```
		- 인접 리스트가 가지는 장점, 메모리 효율

### 그래프 탐색

그래프 탐색이란 그래프의 모든 정점을 체계적으로 방문하는 과정을 의미합니다.

그래프 탐색 알고리즘에는 깊이 우선 탐색과 너비 우선 탐색이 대표적입니다.

>### 깊이 우선 탐색

깊이 우선 탐새은 그래프의 한 정점에서 시작하여

연결된 다른 정점으로 깊게 우선적으로 탐색하는 알고리즘입니다.

우선 한 정점에서 더 이상 방문할 노드가 없을 때까지 깊이 탐색합니다.

그리고 더 이상 방문할 노드가 없으면, 가장 최근에 방문한 이전 노드로 돌아갑니다.

그리고 해당 노드에서 기존에 방문하지 않은 노드가 있는지 확인하고,

있으면 해당 노드에 연결된 가장 깊은 노드까지 다시 탐색합니다.

이 과정을 모든 노드를 방문할 때까지 수행하는것이 깊이 우선 탐색입니다.

깊이 우선 탐색을 표현할 때는 재귀 함수로 구현하는 방법이 대표적입니다.

```cpp

// 목적: DFS는 한 방향으로 가능한 깊이까지 먼저 탐색하는 방식임을 설명
// 동작: 0 → 1 → 3 → 4 → 2

#include <iostream>
#include <vector>
using namespace std;

const int N = 5;
vector<int> graph[N];
bool visited[N];

void DFS(int v) {
    visited[v] = true;
    cout << v << " ";

    for (int u : graph[v]) {
        if (!visited[u]) {
            DFS(u);
        }
    }
}

int main() {
    graph[0] = {1, 2};
    graph[1] = {3};
    graph[3] = {4};

    DFS(0);
    return 0;
}

/*
출력결과
0 1 3 4 2
*/

```
		- 재귀 함수로 구현한 깊이 우선 탐색

>### 너비 우선 탐색

너비 우선 탐색은 가까운 노드를 방문하여, 방문한 노드와 인접한 노드를 큐에 차례대로 넣어

먼저 들어온 순서대로 꺼내 방문하는 방식입니다.

이 과정은 큐의 FIFO를 활욯하면 매우 쉽게 구현이 가능합니다.

너비 우선 탐색에서 찾은 노드의 경로는 최단 경로를 보장합니다.

```cpp
// 목적: BFS는 가까운 노드부터 방문하므로 계층적 탐색이 가능함
// 동작: 0 → 1 → 2 → 3 → 4 순서로 방문됨

#include <iostream>
#include <vector>
#include <queue>
using namespace std;

const int N = 5;
vector<int> graph[N];
bool visited[N];

void BFS(int v) {
    queue<int> Q;
    visited[v] = true;
    Q.push(v);

    while (!Q.empty()) {
        int u = Q.front(); Q.pop();
        cout << u << " ";
        for (int w : graph[u]) {
            if (!visited[w]) {
                visited[w] = true;
                Q.push(w);
            }
        }
    }
}

int main() {
    graph[0] = {1, 2};
    graph[1] = {3};
    graph[2] = {4};

    BFS(0);
    return 0;
}

/*
출력결과
0 1 2 3 4
*/
```
		- 큐를 활용한 너비 우선 탐색 구현

## 백 트래킹

백 트래킹이란. 해당 데이터가 있는 가능성이 없는 경로를 배재하는 방식입니다.

다른 경로를 조기에 배제하게 되면 완전 탐색에 비해 효율성을 크게 향상시킬 수 있습니다.

제약 만족 문제와 같이 주어진 제약 조건을 모두 만족하는 해답을 찾는 문제는

제약 조건을 만족할 때 마다 해당 제약 조건이 아닌 경로를 줄여가면

찾을 데이터의 양이 줄어들어 효율적으로 조사할 수 있게 됩니다.


### 유망 함수

백트래킹을 핵심은 답이 될 가능성이 있는지 판단하는 것입니다.

이 가능성을 판단하는 함수가 유망 함수입니다.

유망 함수는 제약 조건 검사기 역할을 하며, 현재까지의 부분 해답이 

문제의 제약 조건을 만족할 가능성이 있는지를 검사합니다.

해답이 없다고 판단되는 경우, false를 반환하여

해당 경로를 더 이상 탐색하지 않아도 된다고 판단할 수 있습니다.

예를 들어 1,2,3,4 중 2 숫자를 포함하여 7이상인 수를 만들때

1과 2로는 화라고 합니다.

```cpp

/*
  [목적 및 동작]
  이 예시 코드는 주어진 Dijkstra(G, s) 알고리즘의 개념을 실제 C++ 코드로 구현하여,
  특정 그래프에서 최단거리를 구하는 방법을 보여줍니다.
  인접 리스트를 활용한 전형적인 Dijkstra 예시이며,
  우선순위 큐(priority_queue)를 사용하여 최단 거리를 빠르게 추출합니다.
*/

#include <iostream>
#include <vector>
#include <queue>
#include <limits>
using namespace std;

static const int INF = numeric_limits<int>::max();

int main() {
    // 그래프 (u -> v, w) : u에서 v로 가중치 w
    // 노드 개수 N, 간선 개수 E
    int N = 5, E = 6;
    vector<vector<pair<int, int>>> graph(N + 1);
    bool visited[5] = { 0 };

    // 간선 정보 삽입 (무방향 그래프라고 가정)
    graph[1].push_back({ 2, 2 });
    graph[2].push_back({ 1, 2 });
    graph[1].push_back({ 3, 3 });
    graph[3].push_back({ 1, 3 });
    graph[2].push_back({ 4, 4 });
    graph[4].push_back({ 2, 4 });
    graph[2].push_back({ 5, 5 });
    graph[5].push_back({ 2, 5 });
    graph[4].push_back({ 5, 1 });
    graph[5].push_back({ 4, 1 });

    // dist 배열 초기화
    vector<int> dist(N + 1, INF);

    // 시작 정점 s
    int s = 1;
    dist[s] = 0;

    // 우선순위 큐 (최소 힙)
    // pair<거리, 정점> 형태로 삽입
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    pq.push({ dist[s], s });

    while (!pq.empty()) {
        auto [currentDist, u] = pq.top();
        pq.pop();

        // 현재 꺼낸 거리 > 이미 기록된 거리면 무시
        if (visited[u]) continue;
        
        visited[u] = true;

        // u와 인접한 각 정점 v에 대해 거리 갱신 시도
        for (auto& edge : graph[u]) {
            int v = edge.first;
            int w = edge.second;
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({ dist[v], v });
            }
        }
    }

    // 결과 출력
    for (int i = 1; i <= N; i++) {
        if (dist[i] == INF) {
            cout << "정점 " << i << "까지의 최단거리는 도달 불가능" << endl;
        }
        else {
            cout << "정점 " << i << "까지의 최단거리: " << dist[i] << endl;
        }
    }

    /*
    출력결과
    정점 1까지의 최단거리: 0
    정점 2까지의 최단거리: 2
    정점 3까지의 최단거리: 3
    정점 4까지의 최단거리: 6
    정점 5까지의 최단거리: 5
    */
    return 0;
}
```


#그래프 #백트래킹 #내일배움캠프 #언리얼5기
